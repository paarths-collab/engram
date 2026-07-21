//! Task-adaptive context routing and payload budgeting.
//!
//! This layer is deliberately deterministic. It decides how much evidence a
//! task needs, which evidence sources are useful, and how much context Engram
//! may return. The coding agent remains responsible for reasoning.

use engram_domain::{EvidencePacket, ReviewComment};
use serde::Serialize;
use std::collections::HashSet;

const RESPONSE_OVERHEAD_CHARS: usize = 700;
const MAX_REVIEW_BODY_CHARS: usize = 900;
const MAX_TASK_CHARS: usize = 6_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    BugFix,
    Feature,
    Refactor,
    Test,
    Documentation,
    Investigation,
    Security,
    General,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskProfile {
    pub kind: TaskKind,
    pub focus: &'static [&'static str],
    pub evidence_limit: usize,
    pub review_limit: usize,
    pub max_context_chars: usize,
    pub min_relative_score: f32,
    pub prefer_tests: bool,
    pub prefer_docs: bool,
}

impl TaskProfile {
    pub fn classify(task: &str) -> Self {
        let task = task.to_lowercase();
        let tokens: HashSet<&str> = task
            .split(|character: char| !character.is_alphanumeric() && character != '_')
            .filter(|token| !token.is_empty())
            .collect();
        let kind = if contains_any(
            &task,
            &tokens,
            &[
                "security",
                "vulnerability",
                "exploit",
                "authorization",
                "authentication",
                "permission",
                "secret",
                "injection",
                "xss",
                "csrf",
            ],
        ) {
            TaskKind::Security
        } else if contains_any(
            &task,
            &tokens,
            &[
                "bug",
                "fix",
                "broken",
                "failure",
                "failing",
                "regression",
                "crash",
                "incorrect",
                "error",
            ],
        ) {
            TaskKind::BugFix
        } else if contains_any(
            &task,
            &tokens,
            &[
                "investigate",
                "diagnose",
                "debug",
                "root cause",
                "why does",
                "understand",
            ],
        ) {
            TaskKind::Investigation
        } else if contains_any(
            &task,
            &tokens,
            &[
                "refactor",
                "restructure",
                "cleanup",
                "clean up",
                "rename",
                "move",
            ],
        ) {
            TaskKind::Refactor
        } else if contains_any(
            &task,
            &tokens,
            &[
                "test",
                "coverage",
                "assertion",
                "fixture",
                "mock",
                "integration spec",
            ],
        ) {
            TaskKind::Test
        } else if contains_any(
            &task,
            &tokens,
            &[
                "documentation",
                "document",
                "docs",
                "readme",
                "changelog",
                "comment",
                "guide",
            ],
        ) {
            TaskKind::Documentation
        } else if contains_any(
            &task,
            &tokens,
            &[
                "add",
                "implement",
                "create",
                "build",
                "support",
                "feature",
                "introduce",
            ],
        ) {
            TaskKind::Feature
        } else {
            TaskKind::General
        };
        Self::for_kind(kind)
    }

    fn for_kind(kind: TaskKind) -> Self {
        match kind {
            TaskKind::BugFix => Self {
                kind,
                focus: &["existing_code", "tests", "review_history"],
                evidence_limit: 7,
                review_limit: 3,
                max_context_chars: 7_000,
                min_relative_score: 0.35,
                prefer_tests: true,
                prefer_docs: false,
            },
            TaskKind::Feature => Self {
                kind,
                focus: &["existing_code", "reusable_symbols", "review_history"],
                evidence_limit: 8,
                review_limit: 2,
                max_context_chars: 7_500,
                min_relative_score: 0.38,
                prefer_tests: false,
                prefer_docs: false,
            },
            TaskKind::Refactor => Self {
                kind,
                focus: &["symbols", "module_structure", "tests"],
                evidence_limit: 8,
                review_limit: 1,
                max_context_chars: 6_500,
                min_relative_score: 0.42,
                prefer_tests: true,
                prefer_docs: false,
            },
            TaskKind::Test => Self {
                kind,
                focus: &["tests", "tested_symbols", "fixtures"],
                evidence_limit: 6,
                review_limit: 1,
                max_context_chars: 5_500,
                min_relative_score: 0.38,
                prefer_tests: true,
                prefer_docs: false,
            },
            TaskKind::Documentation => Self {
                kind,
                focus: &["documentation", "public_symbols"],
                evidence_limit: 5,
                review_limit: 0,
                max_context_chars: 4_000,
                min_relative_score: 0.45,
                prefer_tests: false,
                prefer_docs: true,
            },
            TaskKind::Investigation => Self {
                kind,
                focus: &["existing_code", "tests", "review_history", "related_paths"],
                evidence_limit: 9,
                review_limit: 4,
                max_context_chars: 8_500,
                min_relative_score: 0.28,
                prefer_tests: true,
                prefer_docs: false,
            },
            TaskKind::Security => Self {
                kind,
                focus: &["existing_code", "tests", "review_history", "rules"],
                evidence_limit: 9,
                review_limit: 5,
                max_context_chars: 9_000,
                min_relative_score: 0.30,
                prefer_tests: true,
                prefer_docs: false,
            },
            TaskKind::General => Self {
                kind,
                focus: &["existing_code", "symbols"],
                evidence_limit: 6,
                review_limit: 1,
                max_context_chars: 5_500,
                min_relative_score: 0.42,
                prefer_tests: false,
                prefer_docs: false,
            },
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ContextSelection {
    pub evidence: Vec<EvidencePacket>,
    pub past_reviews: Vec<ReviewComment>,
    pub used_context_chars: usize,
    pub approximate_tokens: usize,
    pub truncated: bool,
    pub weak_candidates_removed: usize,
}

pub fn select_context(
    mut evidence: Vec<EvidencePacket>,
    reviews: Vec<ReviewComment>,
    profile: &TaskProfile,
) -> ContextSelection {
    let original_evidence_len = evidence.len();
    if let Some(top_score) = evidence
        .iter()
        .map(|packet| packet.score)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
    {
        let threshold = top_score * profile.min_relative_score;
        evidence.retain(|packet| packet.score >= threshold);
    }
    let weak_candidates_removed = original_evidence_len.saturating_sub(evidence.len());
    evidence.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    evidence.truncate(profile.evidence_limit);

    let payload_budget = profile
        .max_context_chars
        .saturating_sub(RESPONSE_OVERHEAD_CHARS);
    let mut selected_evidence = Vec::new();
    let mut selected_reviews = Vec::new();
    let mut used = 0usize;
    let mut truncated = weak_candidates_removed > 0;

    for packet in evidence {
        let cost = json_chars(&packet);
        if used + cost > payload_budget {
            truncated = true;
            break;
        }
        used += cost;
        selected_evidence.push(packet);
    }

    let mut seen_reviews = HashSet::new();
    for mut review in reviews {
        if selected_reviews.len() >= profile.review_limit {
            truncated = true;
            break;
        }
        let key = (review.pr_number, review.path.clone(), review.body.clone());
        if !seen_reviews.insert(key) {
            continue;
        }
        if !selected_evidence
            .iter()
            .any(|packet| packet.path == review.path)
        {
            continue;
        }
        review.body = truncate_chars(&review.body, MAX_REVIEW_BODY_CHARS);
        let cost = json_chars(&review);
        if used + cost > payload_budget {
            truncated = true;
            break;
        }
        used += cost;
        selected_reviews.push(review);
    }

    ContextSelection {
        evidence: selected_evidence,
        past_reviews: selected_reviews,
        used_context_chars: used + RESPONSE_OVERHEAD_CHARS,
        approximate_tokens: (used + RESPONSE_OVERHEAD_CHARS).div_ceil(4),
        truncated,
        weak_candidates_removed,
    }
}

pub fn bound_task(task: &str) -> (String, bool) {
    if task.chars().count() <= MAX_TASK_CHARS {
        return (task.to_owned(), false);
    }
    (task.chars().take(MAX_TASK_CHARS).collect(), true)
}

fn contains_any(text: &str, tokens: &HashSet<&str>, needles: &[&str]) -> bool {
    needles.iter().any(|needle| {
        if needle.contains(' ') {
            text.contains(needle)
        } else {
            tokens.contains(needle)
        }
    })
}

fn json_chars<T: Serialize>(value: &T) -> usize {
    serde_json::to_string(value)
        .map(|json| json.chars().count())
        .unwrap_or(0)
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let mut truncated: String = text.chars().take(max_chars.saturating_sub(16)).collect();
    truncated.push_str("… [truncated]");
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_domain::{EvidenceKind, EvidencePacket};

    fn packet(id: &str, score: f32, snippet_len: usize) -> EvidencePacket {
        EvidencePacket {
            id: id.to_owned(),
            kind: EvidenceKind::ExistingCode,
            title: id.to_owned(),
            path: format!("src/{id}.rs"),
            symbol: None,
            snippet: Some("x".repeat(snippet_len)),
            score,
            signals: vec!["bm25".to_owned()],
        }
    }

    #[test]
    fn classifies_task_before_selecting_evidence() {
        assert_eq!(
            TaskProfile::classify("fix the failing refund test").kind,
            TaskKind::BugFix
        );
        assert_eq!(
            TaskProfile::classify("document the public webhook API").kind,
            TaskKind::Documentation
        );
        assert_eq!(
            TaskProfile::classify("investigate why billing is slow").kind,
            TaskKind::Investigation
        );
    }

    #[test]
    fn security_has_priority_over_feature_words() {
        assert_eq!(
            TaskProfile::classify("add authorization checks to admin routes").kind,
            TaskKind::Security
        );
    }

    #[test]
    fn classification_uses_words_not_substrings() {
        assert_ne!(
            TaskProfile::classify("update the latest dependency").kind,
            TaskKind::Test
        );
        assert_ne!(
            TaskProfile::classify("change contest behavior").kind,
            TaskKind::Test
        );
    }

    #[test]
    fn removes_weak_candidates_and_respects_budget() {
        let profile = TaskProfile::for_kind(TaskKind::General);
        let evidence = vec![
            packet("strong", 1.0, 400),
            packet("useful", 0.6, 400),
            packet("weak", 0.1, 400),
            packet("oversized", 0.9, 10_000),
        ];
        let selected = select_context(evidence, Vec::new(), &profile);
        assert!(selected.evidence.iter().all(|packet| packet.id != "weak"));
        assert!(selected.used_context_chars <= profile.max_context_chars);
        assert!(selected.truncated);
    }

    #[test]
    fn truncates_review_bodies_on_utf8_boundaries() {
        let text = "🦀".repeat(1_000);
        let truncated = truncate_chars(&text, 100);
        assert!(truncated.chars().count() <= 100);
        assert!(truncated.ends_with("[truncated]"));
    }

    #[test]
    fn bounds_oversized_task_on_utf8_boundaries() {
        let (task, truncated) = bound_task(&"🦀".repeat(MAX_TASK_CHARS + 1));
        assert!(truncated);
        assert_eq!(task.chars().count(), MAX_TASK_CHARS);
    }

    #[test]
    fn reviews_must_belong_to_selected_evidence() {
        let profile = TaskProfile::for_kind(TaskKind::BugFix);
        let reviews = vec![ReviewComment {
            pr_number: 7,
            pr_title: "Old change".to_owned(),
            pr_merged: true,
            path: "src/unrelated.rs".to_owned(),
            line: Some(10),
            body: "This review is unrelated to the selected evidence.".to_owned(),
            author: "reviewer".to_owned(),
        }];
        let selected = select_context(vec![packet("strong", 1.0, 100)], reviews, &profile);
        assert!(selected.past_reviews.is_empty());
    }
}
