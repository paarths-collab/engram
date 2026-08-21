//! Frozen, labeled evaluation for the honest reuse-assessment contract.

use anyhow::{bail, Context, Result};
use engram_domain::{EvidencePacket, ReuseAssessment, ReuseState};
use engram_repo_map::store::Store;
use engram_retrieval::Engine;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const EXPECTED_SCHEMA_VERSION: u32 = 2;
const EXPECTED_CASES: usize = 100;
const REUSE_LIKELY_PRECISION_GATE: f64 = 0.90;
const RECALL_AT_3_GATE: f64 = 0.80;
const CORRECT_ABSTENTION_GATE: f64 = 0.85;
const CITATION_VALIDITY_GATE: f64 = 1.0;
const FIXED_GIT_TIME: &str = "2000-01-01T00:00:00+00:00";
const STRATA: [&str; 10] = [
    "exact_existing_implementation",
    "same_behavior_different_terminology",
    "renamed_copied_function",
    "partial_duplication",
    "similar_vocabulary_different_behavior",
    "deprecated_implementation",
    "test_helper_vs_production_helper",
    "no_matching_implementation",
    "incomplete_index",
    "conflicting_approved_decisions",
];

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: u32,
    corpora: BTreeMap<String, PathBuf>,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    stratum: String,
    query: String,
    index_profile: IndexProfile,
    expected_state: ReuseState,
    #[serde(default)]
    expected_candidates: Vec<CandidateLabel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
enum IndexProfile {
    Complete,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct CandidateLabel {
    path: String,
    symbol: String,
    start_line: usize,
}

#[derive(Debug)]
struct ObservedCandidate {
    state: ReuseState,
    path: String,
    symbol: Option<String>,
    start_line: Option<usize>,
    citation_valid: bool,
}

#[derive(Debug)]
struct Observation {
    state: ReuseState,
    candidates: Vec<ObservedCandidate>,
    metadata_valid: bool,
    latency: Duration,
}

#[derive(Debug, Default)]
struct Counts {
    reuse_likely_true: usize,
    reuse_likely_total: usize,
    recall_hits: usize,
    recall_total: usize,
    abstention_correct: usize,
    abstention_total: usize,
    citations_valid: usize,
    citations_total: usize,
    states_correct: usize,
    cases_total: usize,
    latencies: Vec<Duration>,
    failures: Failures,
}

#[derive(Debug, Default)]
struct Failures {
    precision: BTreeSet<String>,
    recall: BTreeSet<String>,
    abstention: BTreeSet<String>,
    citation: BTreeSet<String>,
    state: BTreeSet<String>,
    contract: BTreeSet<String>,
}

#[derive(Debug)]
struct Metrics {
    reuse_likely_precision: f64,
    recall_at_3: f64,
    correct_abstention: f64,
    citation_validity: f64,
    state_accuracy: f64,
    query_p95: Duration,
    counts: Counts,
}

pub fn run(args: &[String]) -> Result<()> {
    let options = Options::parse(args)?;
    let cases_path = options.cases.unwrap_or_else(default_cases_path);
    let (manifest, benchmark_root) = load_manifest(&cases_path)?;
    validate_manifest(&manifest, &benchmark_root)?;

    eprintln!(
        "[evals:reuse] materializing {} isolated frozen corpus profiles from {}",
        manifest.corpora.values().collect::<BTreeSet<_>>().len(),
        benchmark_root.display()
    );

    let fixture_keys: BTreeSet<(String, IndexProfile)> = manifest
        .cases
        .iter()
        .map(|case| (case.stratum.clone(), case.index_profile))
        .collect();
    let mut fixtures = BTreeMap::new();
    let mut snapshots_by_corpus: BTreeMap<PathBuf, String> = BTreeMap::new();
    for (stratum, index_profile) in fixture_keys {
        let relative = manifest
            .corpora
            .get(&stratum)
            .with_context(|| format!("missing corpus profile for {stratum}"))?;
        let corpus = benchmark_root.join(relative);
        let eager_tier1_limit = match index_profile {
            IndexProfile::Complete => usize::MAX,
            IndexProfile::Incomplete => 0,
        };
        let fixture = IndexedFixture::new(&corpus, eager_tier1_limit)?;
        if let Some(existing) = snapshots_by_corpus.get(relative) {
            if existing != &fixture.snapshot_sha {
                bail!(
                    "corpus {} did not produce a deterministic snapshot across index profiles",
                    relative.display()
                );
            }
        } else {
            snapshots_by_corpus.insert(relative.clone(), fixture.snapshot_sha.clone());
        }
        fixtures.insert((stratum, index_profile), fixture);
    }

    let mut counts = Counts::default();
    for case in &manifest.cases {
        let fixture = fixtures
            .get_mut(&(case.stratum.clone(), case.index_profile))
            .with_context(|| format!("missing indexed fixture for {}", case.id))?;
        let started = Instant::now();
        // Contract invariant: exactly one public reuse-assessment call per case.
        let assessment = fixture
            .engine
            .assess_reuse(&mut fixture.store, &case.query)
            .with_context(|| format!("reuse assessment failed for {}", case.id))?;
        let latency = started.elapsed();
        let observation = observe_assessment(case, assessment, fixture.root(), latency);
        counts.record(case, observation);
    }

    let metrics = Metrics::from_counts(counts);
    print_report(&manifest, &snapshots_by_corpus, &metrics);
    if options.check {
        enforce_gates(&metrics)?;
    }
    Ok(())
}

struct Options {
    cases: Option<PathBuf>,
    check: bool,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self> {
        let mut cases = None;
        let mut check = false;
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--check" => check = true,
                "--cases" => {
                    index += 1;
                    let value = args.get(index).context("--cases requires a path")?;
                    cases = Some(PathBuf::from(value));
                }
                unknown => bail!("unknown reuse benchmark argument: {unknown}"),
            }
            index += 1;
        }
        Ok(Self { cases, check })
    }
}

impl Counts {
    fn record(&mut self, case: &Case, observation: Observation) {
        self.cases_total += 1;
        self.latencies.push(observation.latency);

        let mut false_high_confidence = false;
        for candidate in &observation.candidates {
            let relevant = case
                .expected_candidates
                .iter()
                .any(|expected| candidate_matches(candidate, expected));
            if candidate.state == ReuseState::ReuseLikely {
                self.reuse_likely_total += 1;
                // Precision measures candidate relevance, independently of
                // whether the case's top-level state label is conservative.
                // State calibration is reported separately below.
                if relevant {
                    self.reuse_likely_true += 1;
                } else {
                    false_high_confidence = true;
                }
            }
            self.citations_total += 1;
            if candidate.citation_valid {
                self.citations_valid += 1;
            } else {
                self.failures.citation.insert(case.id.clone());
            }
        }
        if false_high_confidence {
            self.failures.precision.insert(case.id.clone());
        }

        for expected in &case.expected_candidates {
            self.recall_total += 1;
            if observation
                .candidates
                .iter()
                .take(3)
                .any(|candidate| candidate_matches(candidate, expected))
            {
                self.recall_hits += 1;
            } else {
                self.failures.recall.insert(case.id.clone());
            }
        }

        if matches!(
            case.expected_state,
            ReuseState::NoEvidence | ReuseState::IndexIncomplete
        ) {
            self.abstention_total += 1;
            if observation.state == case.expected_state && observation.candidates.is_empty() {
                self.abstention_correct += 1;
            } else {
                self.failures.abstention.insert(case.id.clone());
            }
        }

        if observation.state == case.expected_state {
            self.states_correct += 1;
        } else {
            self.failures.state.insert(case.id.clone());
        }
        if observation.candidates.len() > 3 || !observation.metadata_valid {
            self.failures.contract.insert(case.id.clone());
        }
    }
}

impl Metrics {
    fn from_counts(mut counts: Counts) -> Self {
        let query_p95 = percentile_95(&mut counts.latencies);
        Self {
            reuse_likely_precision: ratio(counts.reuse_likely_true, counts.reuse_likely_total),
            recall_at_3: ratio(counts.recall_hits, counts.recall_total),
            correct_abstention: ratio(counts.abstention_correct, counts.abstention_total),
            citation_validity: ratio(counts.citations_valid, counts.citations_total),
            state_accuracy: ratio(counts.states_correct, counts.cases_total),
            query_p95,
            counts,
        }
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn percentile_95(values: &mut [Duration]) -> Duration {
    if values.is_empty() {
        return Duration::ZERO;
    }
    values.sort_unstable();
    let rank = ((values.len() as f64 * 0.95).ceil() as usize).saturating_sub(1);
    values[rank.min(values.len() - 1)]
}

fn candidate_matches(observed: &ObservedCandidate, expected: &CandidateLabel) -> bool {
    observed.path == expected.path
        && observed.symbol.as_deref() == Some(expected.symbol.as_str())
        && observed.start_line == Some(expected.start_line)
}

fn line_defines_symbol(line: &str, symbol: &str) -> bool {
    let needle = format!("fn {symbol}");
    line.match_indices(&needle).any(|(start, _)| {
        let before_ok = start == 0
            || !line[..start]
                .chars()
                .next_back()
                .is_some_and(|ch| ch.is_alphanumeric() || ch == '_');
        let end = start + needle.len();
        let after_ok = line[end..]
            .chars()
            .next()
            .is_none_or(|ch| !ch.is_alphanumeric() && ch != '_');
        before_ok && after_ok
    })
}

fn observe_assessment(
    case: &Case,
    assessment: ReuseAssessment,
    repo: &Path,
    latency: Duration,
) -> Observation {
    let expected_complete = case.index_profile == IndexProfile::Complete;
    let metadata_valid = assessment.index_complete == expected_complete
        && assessment.indexed_files == count_corpus_files(repo);
    let candidates = assessment
        .candidates
        .into_iter()
        .map(|candidate| ObservedCandidate {
            state: candidate.state,
            path: candidate.evidence.path.clone(),
            symbol: candidate.evidence.symbol.clone(),
            start_line: candidate.evidence.start_line,
            citation_valid: citation_is_valid(repo, &candidate.evidence),
        })
        .collect();
    Observation {
        state: assessment.state,
        candidates,
        metadata_valid,
        latency,
    }
}

fn citation_is_valid(repo: &Path, evidence: &EvidencePacket) -> bool {
    let relative = Path::new(&evidence.path);
    if relative.as_os_str().is_empty()
        || evidence.path.contains('\\')
        || relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return false;
    }
    let (Some(symbol), Some(start_line), Some(end_line), Some(snippet)) = (
        evidence.symbol.as_deref(),
        evidence.start_line,
        evidence.end_line,
        evidence.snippet.as_deref(),
    ) else {
        return false;
    };
    if symbol.is_empty()
        || start_line == 0
        || snippet.trim().is_empty()
        || evidence.signals.is_empty()
        || evidence.symbol_kind.is_none()
        || !evidence.score.is_finite()
        || evidence.score < 0.0
        || evidence.signals.iter().collect::<HashSet<_>>().len() != evidence.signals.len()
    {
        return false;
    }
    let Ok(source) = fs::read_to_string(repo.join(relative)) else {
        return false;
    };
    let lines: Vec<&str> = source.lines().collect();
    if end_line < start_line || end_line > lines.len() {
        return false;
    }
    let definition = lines[start_line - 1].trim();
    let snippet_start = snippet.lines().find(|line| !line.trim().is_empty());
    line_defines_symbol(definition, symbol)
        && snippet_start.is_some_and(|line| line.trim() == definition)
}

fn print_report(manifest: &Manifest, snapshots: &BTreeMap<PathBuf, String>, metrics: &Metrics) {
    println!("# Honest reuse retrieval benchmark\n");
    println!("Frozen corpus snapshots:");
    for (profile, snapshot_sha) in snapshots {
        println!("- `{}`: `{snapshot_sha}`", profile.display());
    }
    println!(
        "\nCases: {} ({} strata)\n",
        manifest.cases.len(),
        STRATA.len()
    );
    println!("| metric | result | launch gate |");
    println!("|---|---:|---:|");
    println!(
        "| reuse_likely precision | {:.1}% ({}/{}) | >= 90% |",
        metrics.reuse_likely_precision * 100.0,
        metrics.counts.reuse_likely_true,
        metrics.counts.reuse_likely_total
    );
    println!(
        "| Recall@3 | {:.1}% ({}/{}) | >= 80% |",
        metrics.recall_at_3 * 100.0,
        metrics.counts.recall_hits,
        metrics.counts.recall_total
    );
    println!(
        "| correct abstention | {:.1}% ({}/{}) | >= 85% |",
        metrics.correct_abstention * 100.0,
        metrics.counts.abstention_correct,
        metrics.counts.abstention_total
    );
    println!(
        "| citation validity | {:.1}% ({}/{}) | 100% |",
        metrics.citation_validity * 100.0,
        metrics.counts.citations_valid,
        metrics.counts.citations_total
    );
    println!(
        "| decision-state accuracy | {:.1}% ({}/{}) | informational |",
        metrics.state_accuracy * 100.0,
        metrics.counts.states_correct,
        metrics.counts.cases_total
    );
    println!(
        "| query p95 | {:.1} ms | informational |\n",
        metrics.query_p95.as_secs_f64() * 1000.0
    );

    print_failures("reuse_likely precision", &metrics.counts.failures.precision);
    print_failures("Recall@3", &metrics.counts.failures.recall);
    print_failures("correct abstention", &metrics.counts.failures.abstention);
    print_failures("citation validity", &metrics.counts.failures.citation);
    print_failures("decision-state accuracy", &metrics.counts.failures.state);
    print_failures("response contract", &metrics.counts.failures.contract);
}

fn print_failures(label: &str, cases: &BTreeSet<String>) {
    if cases.is_empty() {
        println!("- {label}: none");
    } else {
        println!(
            "- {label}: {}",
            cases.iter().cloned().collect::<Vec<_>>().join(", ")
        );
    }
}

fn enforce_gates(metrics: &Metrics) -> Result<()> {
    let mut failed = Vec::new();
    if metrics.reuse_likely_precision < REUSE_LIKELY_PRECISION_GATE {
        failed.push("reuse_likely precision < 0.90");
    }
    if metrics.recall_at_3 < RECALL_AT_3_GATE {
        failed.push("Recall@3 < 0.80");
    }
    if metrics.correct_abstention < CORRECT_ABSTENTION_GATE {
        failed.push("correct abstention < 0.85");
    }
    if metrics.citation_validity < CITATION_VALIDITY_GATE {
        failed.push("citation validity < 1.00");
    }
    if failed.is_empty() {
        Ok(())
    } else {
        bail!("reuse benchmark launch gates failed: {}", failed.join("; "))
    }
}

fn default_cases_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/reuse/cases.json")
}

fn load_manifest(path: &Path) -> Result<(Manifest, PathBuf)> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read reuse cases from {}", path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid reuse case manifest {}", path.display()))?;
    let parent = path
        .parent()
        .context("reuse cases path has no parent")?
        .to_path_buf();
    Ok((manifest, parent))
}

fn validate_manifest(manifest: &Manifest, benchmark_root: &Path) -> Result<()> {
    if manifest.schema_version != EXPECTED_SCHEMA_VERSION {
        bail!(
            "unsupported reuse manifest schema {}; expected {}",
            manifest.schema_version,
            EXPECTED_SCHEMA_VERSION
        );
    }
    if manifest.cases.len() != EXPECTED_CASES {
        bail!(
            "reuse manifest must contain {EXPECTED_CASES} cases; found {}",
            manifest.cases.len()
        );
    }

    let expected_strata: BTreeSet<&str> = STRATA.into_iter().collect();
    let configured_strata: BTreeSet<&str> = manifest.corpora.keys().map(String::as_str).collect();
    if configured_strata != expected_strata {
        bail!("corpus profiles must map exactly the ten requested strata");
    }
    for (stratum, relative) in &manifest.corpora {
        if relative.as_os_str().is_empty()
            || relative.is_absolute()
            || relative.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            bail!("corpus profile {stratum} must be a relative child directory");
        }
        let corpus = benchmark_root.join(relative);
        if !corpus.is_dir() {
            bail!(
                "corpus profile {stratum} does not exist at {}",
                corpus.display()
            );
        }
    }

    let mut ids = HashSet::new();
    let mut strata: BTreeMap<&str, usize> = BTreeMap::new();
    for case in &manifest.cases {
        if case.id.trim().is_empty() || !ids.insert(&case.id) {
            bail!("reuse case IDs must be non-empty and unique: {}", case.id);
        }
        if case.query.trim().is_empty() {
            bail!("reuse case {} has an empty query", case.id);
        }
        *strata.entry(case.stratum.as_str()).or_default() += 1;
        let positive = matches!(
            case.expected_state,
            ReuseState::ReuseLikely | ReuseState::PossibleReuse
        );
        if (case.index_profile == IndexProfile::Incomplete)
            != (case.expected_state == ReuseState::IndexIncomplete)
        {
            bail!(
                "reuse case {} must pair the incomplete profile with index_incomplete",
                case.id
            );
        }
        if positive != !case.expected_candidates.is_empty() {
            bail!(
                "reuse case {} must label candidates exactly when it expects reuse",
                case.id
            );
        }
        let corpus = benchmark_root.join(
            manifest
                .corpora
                .get(&case.stratum)
                .with_context(|| format!("case {} has no corpus profile", case.id))?,
        );
        let mut candidate_identities = HashSet::new();
        for candidate in &case.expected_candidates {
            let path = Path::new(&candidate.path);
            if candidate.symbol.trim().is_empty()
                || !candidate.symbol.chars().enumerate().all(|(index, ch)| {
                    ch == '_' || ch.is_ascii_alphanumeric() && (index > 0 || !ch.is_ascii_digit())
                })
                || candidate.start_line == 0
                || candidate.path.contains('\\')
                || path.is_absolute()
                || path.components().any(|part| {
                    matches!(
                        part,
                        Component::ParentDir | Component::RootDir | Component::Prefix(_)
                    )
                })
                || !corpus.join(path).is_file()
            {
                bail!("reuse case {} has an invalid candidate label", case.id);
            }
            if !candidate_identities.insert((
                candidate.path.as_str(),
                candidate.symbol.as_str(),
                candidate.start_line,
            )) {
                bail!("reuse case {} has a duplicate candidate label", case.id);
            }
            let source = fs::read_to_string(corpus.join(path)).with_context(|| {
                format!("failed to read candidate source for reuse case {}", case.id)
            })?;
            let Some(line) = source.lines().nth(candidate.start_line - 1) else {
                bail!(
                    "reuse case {} candidate line {} is outside {}",
                    case.id,
                    candidate.start_line,
                    candidate.path
                );
            };
            if !line_defines_symbol(line, &candidate.symbol) {
                bail!(
                    "reuse case {} label does not resolve to function {} at {}:{}",
                    case.id,
                    candidate.symbol,
                    candidate.path,
                    candidate.start_line
                );
            }
        }
    }
    for stratum in STRATA {
        if strata.get(stratum).copied() != Some(10) {
            bail!("stratum {stratum} must contain exactly 10 cases");
        }
    }
    if strata.len() != STRATA.len() {
        bail!("reuse manifest contains an unknown stratum");
    }
    Ok(())
}

fn count_corpus_files(root: &Path) -> usize {
    fn count(path: &Path) -> usize {
        let Ok(entries) = fs::read_dir(path) else {
            return 0;
        };
        entries
            .filter_map(|entry| entry.ok())
            .map(|entry| {
                let path = entry.path();
                if entry.file_name() == ".git" || entry.file_name() == ".engram" {
                    0
                } else if path.is_dir() {
                    count(&path)
                } else if path.is_file() {
                    1
                } else {
                    0
                }
            })
            .sum()
    }
    count(root)
}

struct IndexedFixture {
    store: Store,
    engine: Engine,
    snapshot_sha: String,
    // Keep the temp repo last so the SQLite store is dropped before Windows
    // removes the directory.
    repo: TempGitRepo,
}

impl IndexedFixture {
    fn new(corpus: &Path, eager_tier1_limit: usize) -> Result<Self> {
        let repo = TempGitRepo::materialize(corpus)?;
        engram_repo_map::index_repo(repo.root(), eager_tier1_limit)?;
        let mut store = Store::open(repo.root())?;
        let engine = Engine::build(repo.root(), &mut store)?;
        let snapshot_sha = git_output(repo.root(), &["rev-parse", "HEAD"])?;
        Ok(Self {
            store,
            engine,
            snapshot_sha,
            repo,
        })
    }

    fn root(&self) -> &Path {
        self.repo.root()
    }
}

struct TempGitRepo {
    root: PathBuf,
}

impl TempGitRepo {
    fn materialize(corpus: &Path) -> Result<Self> {
        let root = loop {
            let serial = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let candidate = std::env::temp_dir()
                .join(format!("engram-reuse-eval-{}-{serial}", std::process::id()));
            match fs::create_dir(&candidate) {
                Ok(()) => break candidate,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).context("failed to create reuse benchmark temp dir")
                }
            }
        };
        if let Err(error) = copy_tree(corpus, &root).and_then(|()| initialize_git(&root)) {
            let _ = fs::remove_dir_all(&root);
            return Err(error);
        }
        Ok(Self { root })
    }

    fn root(&self) -> &Path {
        &self.root
    }
}

impl Drop for TempGitRepo {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    for entry in fs::read_dir(source)
        .with_context(|| format!("failed to read corpus directory {}", source.display()))?
    {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            fs::create_dir(&target)?;
            copy_tree(&entry.path(), &target)?;
        } else if entry.file_type()?.is_file() {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

fn initialize_git(root: &Path) -> Result<()> {
    git_output(root, &["init", "--quiet"])?;
    git_output(root, &["config", "user.name", "Engram Benchmark"])?;
    git_output(root, &["config", "user.email", "benchmark@engram.invalid"])?;
    git_output(root, &["config", "core.autocrlf", "false"])?;
    git_output(root, &["add", "--all"])?;
    git_output_with_fixed_time(root, &["commit", "--quiet", "-m", "frozen reuse corpus"])?;
    Ok(())
}

fn git_output(root: &Path, args: &[&str]) -> Result<String> {
    git_command(root, args, false)
}

fn git_output_with_fixed_time(root: &Path, args: &[&str]) -> Result<String> {
    git_command(root, args, true)
}

fn git_command(root: &Path, args: &[&str], fixed_time: bool) -> Result<String> {
    let mut command = Command::new("git");
    command.args(args).current_dir(root);
    if fixed_time {
        command
            .env("GIT_AUTHOR_DATE", FIXED_GIT_TIME)
            .env("GIT_COMMITTER_DATE", FIXED_GIT_TIME);
    }
    let output = command
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_default() -> (Manifest, PathBuf) {
        load_manifest(&default_cases_path()).expect("default reuse manifest")
    }

    #[test]
    fn frozen_manifest_has_all_requested_strata() {
        let (manifest, benchmark_root) = load_default();
        validate_manifest(&manifest, &benchmark_root).unwrap();
        assert_eq!(manifest.cases.len(), 100);
        assert_eq!(manifest.corpora.len(), 10);
    }

    #[test]
    fn fixed_git_time_produces_the_same_snapshot_for_every_profile() {
        let (manifest, benchmark_root) = load_default();
        for corpus in manifest.corpora.values().collect::<BTreeSet<_>>() {
            let corpus = benchmark_root.join(corpus);
            let first = TempGitRepo::materialize(&corpus).unwrap();
            let second = TempGitRepo::materialize(&corpus).unwrap();
            assert_eq!(
                git_output(first.root(), &["rev-parse", "HEAD"]).unwrap(),
                git_output(second.root(), &["rev-parse", "HEAD"]).unwrap(),
                "snapshot changed for {}",
                corpus.display()
            );
        }
    }

    #[test]
    fn scoring_separates_precision_recall_abstention_and_state() {
        let case = Case {
            id: "case".to_owned(),
            stratum: STRATA[0].to_owned(),
            query: "query".to_owned(),
            index_profile: IndexProfile::Complete,
            expected_state: ReuseState::ReuseLikely,
            expected_candidates: vec![CandidateLabel {
                path: "src/exact.rs".to_owned(),
                symbol: "retry_with_backoff".to_owned(),
                start_line: 1,
            }],
        };
        let observation = Observation {
            state: ReuseState::ReuseLikely,
            candidates: vec![ObservedCandidate {
                state: ReuseState::ReuseLikely,
                path: "src/exact.rs".to_owned(),
                symbol: Some("retry_with_backoff".to_owned()),
                start_line: Some(1),
                citation_valid: true,
            }],
            metadata_valid: true,
            latency: Duration::from_millis(5),
        };
        let mut counts = Counts::default();
        counts.record(&case, observation);
        let metrics = Metrics::from_counts(counts);
        assert_eq!(metrics.reuse_likely_precision, 1.0);
        assert_eq!(metrics.recall_at_3, 1.0);
        assert_eq!(metrics.citation_validity, 1.0);
        assert_eq!(metrics.state_accuracy, 1.0);
    }

    #[test]
    fn relevant_likely_candidate_is_precise_even_when_state_label_is_possible() {
        let case = Case {
            id: "conservative-label".to_owned(),
            stratum: STRATA[1].to_owned(),
            query: "query".to_owned(),
            index_profile: IndexProfile::Complete,
            expected_state: ReuseState::PossibleReuse,
            expected_candidates: vec![CandidateLabel {
                path: "src/synonyms.rs".to_owned(),
                symbol: "delay_and_repeat".to_owned(),
                start_line: 1,
            }],
        };
        let observation = Observation {
            state: ReuseState::ReuseLikely,
            candidates: vec![ObservedCandidate {
                state: ReuseState::ReuseLikely,
                path: "src/synonyms.rs".to_owned(),
                symbol: Some("delay_and_repeat".to_owned()),
                start_line: Some(1),
                citation_valid: true,
            }],
            metadata_valid: true,
            latency: Duration::ZERO,
        };
        let mut counts = Counts::default();
        counts.record(&case, observation);
        let metrics = Metrics::from_counts(counts);
        assert_eq!(metrics.reuse_likely_precision, 1.0);
        assert_eq!(metrics.state_accuracy, 0.0);
    }

    #[test]
    fn symbol_definition_validation_uses_exact_identity_and_line() {
        assert!(line_defines_symbol(
            "pub fn retry_with_backoff() {}",
            "retry_with_backoff"
        ));
        assert!(!line_defines_symbol(
            "pub fn retry_with_backoff_extra() {}",
            "retry_with_backoff"
        ));
        assert!(!line_defines_symbol(
            "// retry_with_backoff",
            "retry_with_backoff"
        ));
    }

    #[test]
    fn p95_uses_nearest_rank() {
        let mut values: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        assert_eq!(percentile_95(&mut values), Duration::from_millis(95));
    }
}
