//! engram-connectors-github: ingest merged pull requests, their changed files,
//! and review comments from the GitHub REST API into the store.
//!
//! Raw evidence only — review comments are stored verbatim, never summarized.
//! Uses a blocking `reqwest` client (GitHub is just HTTP + JSON); a custom CA
//! bundle is honored via `ENGRAM_CA_BUNDLE`/`SSL_CERT_FILE` for proxied envs,
//! and `HTTPS_PROXY` is read from the environment automatically.

use anyhow::{bail, Context, Result};
use engram_repo_map::store::Store;
use serde::Deserialize;

const API_BASE: &str = "https://api.github.com";

/// A raw review comment: `(path, line, body, author)`.
pub type RawComment = (String, Option<i64>, String, String);

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
    merged_at: Option<String>,
    #[serde(default)]
    user: Option<UserJson>,
}

#[derive(Debug, Deserialize)]
struct UserJson {
    #[serde(default)]
    login: String,
}

#[derive(Debug, Deserialize)]
struct FileJson {
    filename: String,
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
}

/// A merged/closed PR summary.
pub struct Pr {
    pub number: i64,
    pub title: String,
    pub body: String,
    pub merged: bool,
    pub author: String,
}

#[derive(Debug, Default)]
pub struct IngestStats {
    pub pull_requests: usize,
    pub files: usize,
    pub review_comments: usize,
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
        Ok(raw
            .into_iter()
            .take(limit)
            .map(|p| Pr {
                number: p.number,
                title: p.title,
                body: p.body.unwrap_or_default(),
                merged: p.merged_at.is_some(),
                author: p.user.map(|u| u.login).unwrap_or_default(),
            })
            .collect())
    }

    /// Changed file paths for a PR.
    pub fn pr_files(&self, number: i64) -> Result<Vec<String>> {
        let text = self.get(&format!(
            "/repos/{}/{}/pulls/{number}/files?per_page=100",
            self.owner, self.repo
        ))?;
        let raw: Vec<FileJson> = serde_json::from_str(&text)?;
        Ok(raw.into_iter().map(|f| f.filename).collect())
    }

    /// Raw review comments for a PR (path, line, body, author).
    pub fn pr_review_comments(&self, number: i64) -> Result<Vec<RawComment>> {
        let text = self.get(&format!(
            "/repos/{}/{}/pulls/{number}/comments?per_page=100",
            self.owner, self.repo
        ))?;
        parse_comments(&text)
    }
}

/// Parse a GitHub review-comments JSON payload into `(path, line, body, author)`.
/// Pure function so it can be unit-tested without network.
pub fn parse_comments(json: &str) -> Result<Vec<RawComment>> {
    let raw: Vec<CommentJson> = serde_json::from_str(json)?;
    Ok(raw
        .into_iter()
        .map(|c| {
            (
                c.path,
                c.line.or(c.original_line),
                c.body,
                c.user.map(|u| u.login).unwrap_or_default(),
            )
        })
        .collect())
}

/// Ingest up to `limit` closed PRs (with files and review comments) into the store.
pub fn ingest(store: &mut Store, client: &GitHubClient, limit: usize) -> Result<IngestStats> {
    let prs = client.recent_closed_prs(limit)?;
    let mut stats = IngestStats::default();
    for pr in &prs {
        store.upsert_pull_request(pr.number, &pr.title, &pr.body, pr.merged, &pr.author)?;
        stats.pull_requests += 1;

        let files = client.pr_files(pr.number)?;
        stats.files += files.len();
        store.replace_pr_files(pr.number, &files)?;

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
        assert_eq!(out[0].0, "src/billing/cancel.rs");
        assert_eq!(out[0].1, Some(42));
        assert_eq!(out[0].3, "alice");
        // falls back to original_line when line is null
        assert_eq!(out[1].1, Some(10));
    }
}
