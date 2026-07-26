//! engram-connectors-github: ingest merged pull requests, their changed files,
//! and review comments from the GitHub REST API into the store.
//!
//! Raw evidence only — review comments are stored verbatim, never summarized.
//! Uses a blocking `reqwest` client (GitHub is just HTTP + JSON); a custom CA
//! bundle is honored via `ENGRAM_CA_BUNDLE`/`SSL_CERT_FILE` for proxied envs,
//! and `HTTPS_PROXY` is read from the environment automatically.

use anyhow::{bail, Context, Result};
use engram_domain::{IngestedComment, PrCommit, PrFileChange};
use engram_repo_map::store::Store;
use serde::Deserialize;

const API_BASE: &str = "https://api.github.com";

/// Resolve a GitHub token from the environment.
pub fn token_from_env() -> Option<String> {
    ["ENGRAM_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"]
        .iter()
        .find_map(|k| std::env::var(k).ok().filter(|v| !v.is_empty()))
}

/// Parse an `owner/repo` slug out of a git remote URL (https, ssh, or a
/// proxy path form like `http://host/git/owner/repo`).
pub fn parse_repo_slug(remote: &str) -> Option<(String, String)> {
    let s = remote.trim().trim_end_matches(".git");
    let tail = if let Some(rest) = s.split_once("github.com").map(|x| x.1) {
        rest.trim_start_matches([':', '/'])
    } else {
        // proxy/self-hosted form: take the last two path segments
        s
    };
    let segs: Vec<&str> = tail.rsplit('/').take(2).collect();
    if segs.len() == 2 && !segs[0].is_empty() && !segs[1].is_empty() {
        Some((segs[1].to_string(), segs[0].to_string()))
    } else {
        None
    }
}

pub struct GitHubClient {
    client: reqwest::blocking::Client,
    token: String,
    owner: String,
    repo: String,
}

#[derive(Debug, Deserialize)]
struct PrJson {
    number: i64,
    #[serde(default)]
    title: String,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    merged_at: Option<String>,
    #[serde(default)]
    user: Option<UserJson>,
    #[serde(default)]
    base: Option<RefJson>,
    #[serde(default)]
    head: Option<RefJson>,
}

/// The `base`/`head` object on a PR — only its SHA is of interest.
#[derive(Debug, Deserialize)]
struct RefJson {
    #[serde(default)]
    sha: String,
}

#[derive(Debug, Deserialize)]
struct UserJson {
    #[serde(default)]
    login: String,
}

#[derive(Debug, Deserialize)]
struct FileJson {
    filename: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    additions: i64,
    #[serde(default)]
    deletions: i64,
    /// Absent for binary files and diffs GitHub considers too large.
    #[serde(default)]
    patch: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CommentJson {
    path: String,
    #[serde(default)]
    line: Option<i64>,
    #[serde(default)]
    original_line: Option<i64>,
    #[serde(default)]
    body: String,
    #[serde(default)]
    user: Option<UserJson>,
    #[serde(default)]
    created_at: String,
    #[serde(default)]
    diff_hunk: String,
    #[serde(default)]
    commit_id: String,
    #[serde(default)]
    in_reply_to_id: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct CommitJson {
    #[serde(default)]
    sha: String,
    #[serde(default)]
    commit: Option<CommitDetailJson>,
}

#[derive(Debug, Deserialize)]
struct CommitDetailJson {
    #[serde(default)]
    message: String,
    #[serde(default)]
    author: Option<CommitAuthorJson>,
}

#[derive(Debug, Deserialize)]
struct CommitAuthorJson {
    #[serde(default)]
    name: String,
    #[serde(default)]
    date: String,
}

/// A merged/closed PR summary.
pub struct Pr {
    pub number: i64,
    pub title: String,
    pub body: String,
    pub merged: bool,
    pub author: String,
    /// SHA the PR branched from.
    pub base_sha: String,
    /// SHA at the tip of the PR branch.
    pub head_sha: String,
    pub created_at: String,
    /// Empty when the PR was closed without merging.
    pub merged_at: String,
}

#[derive(Debug, Default)]
pub struct IngestStats {
    pub pull_requests: usize,
    pub files: usize,
    pub review_comments: usize,
    pub commits: usize,
    /// Files whose diff GitHub omitted (binary or oversized).
    pub files_without_patch: usize,
}

impl GitHubClient {
    pub fn new(token: String, owner: String, repo: String) -> Result<Self> {
        let mut builder = reqwest::blocking::Client::builder()
            .user_agent("engram-connectors-github")
            .timeout(std::time::Duration::from_secs(30));
        // Trust an extra CA bundle when running behind a MITM proxy.
        for var in ["ENGRAM_CA_BUNDLE", "SSL_CERT_FILE"] {
            if let Ok(path) = std::env::var(var) {
                if let Ok(pem) = std::fs::read(&path) {
                    if let Ok(certs) = reqwest::Certificate::from_pem_bundle(&pem) {
                        for c in certs {
                            builder = builder.add_root_certificate(c);
                        }
                    }
                }
            }
        }
        Ok(GitHubClient {
            client: builder.build()?,
            token,
            owner,
            repo,
        })
    }

    fn get(&self, path: &str) -> Result<String> {
        let url = format!("{API_BASE}{path}");
        let resp = self
            .client
            .get(&url)
            .header("Accept", "application/vnd.github+json")
            .header("Authorization", format!("Bearer {}", self.token))
            .header("X-GitHub-Api-Version", "2022-11-28")
            .send()
            .with_context(|| format!("request failed: {url}"))?;
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            bail!(
                "GitHub API {status} for {url}: {}",
                text.chars().take(200).collect::<String>()
            );
        }
        Ok(text)
    }

    /// Fetch up to `limit` most-recently-updated closed PRs.
    pub fn recent_closed_prs(&self, limit: usize) -> Result<Vec<Pr>> {
        let per_page = limit.min(100);
        let text = self.get(&format!(
            "/repos/{}/{}/pulls?state=closed&sort=updated&direction=desc&per_page={per_page}",
            self.owner, self.repo
        ))?;
        let raw: Vec<PrJson> = serde_json::from_str(&text)?;
        Ok(raw.into_iter().take(limit).map(parse_pr).collect())
    }

    /// Changed files for a PR, with their diff hunks.
    pub fn pr_files(&self, number: i64) -> Result<Vec<PrFileChange>> {
        let text = self.get(&format!(
            "/repos/{}/{}/pulls/{number}/files?per_page=100",
            self.owner, self.repo
        ))?;
        parse_files(&text)
    }

    /// Commits belonging to a PR, in API order.
    pub fn pr_commits(&self, number: i64) -> Result<Vec<PrCommit>> {
        let text = self.get(&format!(
            "/repos/{}/{}/pulls/{number}/commits?per_page=100",
            self.owner, self.repo
        ))?;
        parse_commits(&text)
    }

    /// Review comments for a PR, with timestamps and anchoring hunks.
    pub fn pr_review_comments(&self, number: i64) -> Result<Vec<IngestedComment>> {
        let text = self.get(&format!(
            "/repos/{}/{}/pulls/{number}/comments?per_page=100",
            self.owner, self.repo
        ))?;
        parse_comments(&text)
    }
}

fn parse_pr(p: PrJson) -> Pr {
    Pr {
        number: p.number,
        title: p.title,
        body: p.body.unwrap_or_default(),
        merged: p.merged_at.is_some(),
        author: p.user.map(|u| u.login).unwrap_or_default(),
        base_sha: p.base.map(|r| r.sha).unwrap_or_default(),
        head_sha: p.head.map(|r| r.sha).unwrap_or_default(),
        created_at: p.created_at.unwrap_or_default(),
        merged_at: p.merged_at.unwrap_or_default(),
    }
}

/// Parse a GitHub pull-request-files payload. Pure so it can be unit-tested
/// without network.
pub fn parse_files(json: &str) -> Result<Vec<PrFileChange>> {
    let raw: Vec<FileJson> = serde_json::from_str(json)?;
    Ok(raw
        .into_iter()
        .map(|f| PrFileChange {
            path: f.filename,
            status: f.status,
            additions: f.additions,
            deletions: f.deletions,
            patch: f.patch.unwrap_or_default(),
        })
        .collect())
}

/// Parse a GitHub pull-request-commits payload, assigning each commit its
/// position in the list. Pure so it can be unit-tested without network.
pub fn parse_commits(json: &str) -> Result<Vec<PrCommit>> {
    let raw: Vec<CommitJson> = serde_json::from_str(json)?;
    Ok(raw
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            let detail = c.commit.unwrap_or(CommitDetailJson {
                message: String::new(),
                author: None,
            });
            let author = detail.author.unwrap_or(CommitAuthorJson {
                name: String::new(),
                date: String::new(),
            });
            PrCommit {
                sha: c.sha,
                message: detail.message,
                author: author.name,
                authored_at: author.date,
                ordinal: i as i64,
            }
        })
        .collect())
}

/// Parse a GitHub review-comments payload. Pure so it can be unit-tested
/// without network.
pub fn parse_comments(json: &str) -> Result<Vec<IngestedComment>> {
    let raw: Vec<CommentJson> = serde_json::from_str(json)?;
    Ok(raw
        .into_iter()
        .map(|c| IngestedComment {
            path: c.path,
            line: c.line.or(c.original_line),
            body: c.body,
            author: c.user.map(|u| u.login).unwrap_or_default(),
            created_at: c.created_at,
            diff_hunk: c.diff_hunk,
            commit_id: c.commit_id,
            in_reply_to: c.in_reply_to_id,
        })
        .collect())
}

/// Ingest up to `limit` closed PRs (with files and review comments) into the store.
pub fn ingest(store: &mut Store, client: &GitHubClient, limit: usize) -> Result<IngestStats> {
    let prs = client.recent_closed_prs(limit)?;
    let mut stats = IngestStats::default();
    for pr in &prs {
        store.upsert_pull_request(
            pr.number,
            &pr.title,
            &pr.body,
            pr.merged,
            &pr.author,
            &pr.base_sha,
            &pr.head_sha,
            &pr.created_at,
            &pr.merged_at,
        )?;
        stats.pull_requests += 1;

        let files = client.pr_files(pr.number)?;
        stats.files += files.len();
        stats.files_without_patch += files.iter().filter(|f| f.patch.is_empty()).count();
        store.replace_pr_files(pr.number, &files)?;

        let commits = client.pr_commits(pr.number)?;
        stats.commits += commits.len();
        store.replace_pr_commits(pr.number, &commits)?;

        let comments = client.pr_review_comments(pr.number)?;
        stats.review_comments += comments.len();
        store.replace_review_comments_for_pr(pr.number, &comments)?;
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_repo_slug_forms() {
        assert_eq!(
            parse_repo_slug("https://github.com/paarths-collab/engram.git"),
            Some(("paarths-collab".to_string(), "engram".to_string()))
        );
        assert_eq!(
            parse_repo_slug("git@github.com:paarths-collab/engram.git"),
            Some(("paarths-collab".to_string(), "engram".to_string()))
        );
        assert_eq!(
            parse_repo_slug("http://local_proxy@127.0.0.1:41729/git/paarths-collab/engram"),
            Some(("paarths-collab".to_string(), "engram".to_string()))
        );
    }

    #[test]
    fn parses_review_comments_with_line_fallback() {
        let json = r#"[
            {"path":"src/billing/cancel.rs","line":42,"body":"reuse SubscriptionStateMachine","user":{"login":"alice"}},
            {"path":"src/webhooks.rs","line":null,"original_line":10,"body":"races with API path","user":{"login":"bob"}}
        ]"#;
        let out = parse_comments(json).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].path, "src/billing/cancel.rs");
        assert_eq!(out[0].line, Some(42));
        assert_eq!(out[0].author, "alice");
        // falls back to original_line when line is null
        assert_eq!(out[1].line, Some(10));
    }

    #[test]
    fn parses_comment_timestamp_hunk_and_thread_position() {
        // The fields that make a comment minable: when it was said, what code
        // it was said about, and whether it starts a thread or replies to one.
        let json = r#"[
            {"path":"src/api.rs","line":7,"body":"wrap this in Context","user":{"login":"alice"},
             "created_at":"2026-05-01T10:00:00Z","commit_id":"abc123",
             "diff_hunk":"@@ -1,3 +1,4 @@\n+    let x = f().unwrap();"},
            {"path":"src/api.rs","line":7,"body":"agreed","user":{"login":"bob"},
             "created_at":"2026-05-01T10:05:00Z","in_reply_to_id":900}
        ]"#;
        let out = parse_comments(json).unwrap();
        assert_eq!(out[0].created_at, "2026-05-01T10:00:00Z");
        assert_eq!(out[0].commit_id, "abc123");
        assert!(out[0].diff_hunk.contains("unwrap()"));
        // A thread root is a change request; a reply is usually discussion.
        assert_eq!(out[0].in_reply_to, None);
        assert_eq!(out[1].in_reply_to, Some(900));
    }

    #[test]
    fn parses_pr_files_with_patch() {
        let json = r#"[
            {"filename":"src/api.rs","status":"modified","additions":3,"deletions":1,
             "patch":"@@ -1,3 +1,5 @@\n-let x = f().unwrap();\n+let x = f().context(\"f\")?;"},
            {"filename":"assets/logo.png","status":"added","additions":0,"deletions":0}
        ]"#;
        let out = parse_files(json).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].path, "src/api.rs");
        assert_eq!(out[0].status, "modified");
        assert_eq!(out[0].additions, 3);
        assert!(out[0].patch.contains("context"));
        // Binary files carry no patch; that must parse, not fail.
        assert_eq!(out[1].patch, "");
    }

    #[test]
    fn parses_pr_commits_in_order() {
        let json = r#"[
            {"sha":"aaa","commit":{"message":"first pass",
             "author":{"name":"alice","date":"2026-05-01T09:00:00Z"}}},
            {"sha":"bbb","commit":{"message":"address review",
             "author":{"name":"alice","date":"2026-05-01T11:00:00Z"}}}
        ]"#;
        let out = parse_commits(json).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].sha, "aaa");
        assert_eq!(out[0].ordinal, 0);
        assert_eq!(out[1].ordinal, 1);
        assert_eq!(out[1].authored_at, "2026-05-01T11:00:00Z");
        assert_eq!(out[1].message, "address review");
    }

    #[test]
    fn parses_pr_shas_and_timestamps() {
        let json = r#"[
            {"number":7,"title":"fix billing","body":"b","created_at":"2026-05-01T08:00:00Z",
             "merged_at":"2026-05-02T08:00:00Z","user":{"login":"alice"},
             "base":{"sha":"base111"},"head":{"sha":"head222"}},
            {"number":8,"title":"abandoned","body":"","created_at":"2026-05-03T08:00:00Z",
             "user":{"login":"bob"},"base":{"sha":"base333"},"head":{"sha":"head444"}}
        ]"#;
        let raw: Vec<PrJson> = serde_json::from_str(json).unwrap();
        let out: Vec<Pr> = raw.into_iter().map(parse_pr).collect();
        assert_eq!(out[0].base_sha, "base111");
        assert_eq!(out[0].head_sha, "head222");
        assert_eq!(out[0].created_at, "2026-05-01T08:00:00Z");
        assert!(out[0].merged);
        // Closed without merging: no merged_at, and merged stays false.
        assert!(!out[1].merged);
        assert_eq!(out[1].merged_at, "");
    }
}
