//! Budgeted, deterministic evidence packs.
//!
//! A tool result is a JSON object holding some arrays of *evidence units*
//! (search hits, review comments, expansion candidates, ...) next to a few
//! scalar fields. [`apply`] turns such a result into a *pack*: it keeps whole
//! units in rank order while the **serialized result as the client receives it**
//! stays within a token budget, and it declares everything it left out in a
//! top-level `pack` object. Nothing is dropped silently.
//!
//! # The token measure
//!
//! No tokenizer is available to this crate, so tokens are *estimated* from the
//! UTF-8 byte length of the exact text the MCP layer returns
//! ([`render`], pretty-printed JSON): `estimate_tokens(text) = ceil(bytes / 3)`,
//! identified as [`TOKEN_ESTIMATOR`] (`"bytes3-v1"`) and echoed in every
//! `pack`. It is an estimator, not a tokenizer, and a response says so by naming
//! it.
//!
//! The divisor is deliberately 3, not the 4 often quoted for English prose.
//! Pretty-printed JSON around source code is the worst case for a tokenizer:
//! punctuation, quotes, brackets, short identifiers and indentation runs
//! frequently cost a token each, so real counts for this content tend to sit
//! near 3 bytes/token or below, while prose reaches 4 or more. Since the budget
//! is a promise not to exceed a size, the estimate must err high: under-counting
//! would let a response overrun, over-counting merely returns slightly less.
//! The function is deterministic and monotonic in the byte length. Changing the
//! divisor means a new estimator id (`bytes3-v2`), never a silent edit.
//!
//! The budget bounds the *decoded* text of the MCP `content[0].text` field, i.e.
//! exactly [`render`]'s output. The JSON-RPC envelope around it (string
//! escaping of quotes and newlines, the `result`/`content` wrapper) is not
//! counted.

use engram_domain::EvidencePacket;
use serde_json::{json, Map, Value};
use std::cmp::Ordering;
use std::collections::HashSet;

/// Identifier of the token estimator, echoed as `pack.estimator`.
pub const TOKEN_ESTIMATOR: &str = "bytes3-v1";
const BYTES_PER_TOKEN: usize = 3;
/// Longest `pack.omitted` list; `omitted_count` still counts every omission.
pub const MAX_OMITTED_LISTED: usize = 20;
/// Under a budget the omitted list is itself bounded by it: one listed entry
/// per this many budget tokens (at most [`MAX_OMITTED_LISTED`]), so the list
/// that apologises for omissions cannot eat the room the evidence needed.
const BUDGET_TOKENS_PER_LISTED_OMISSION: usize = 250;
/// `pack.ordering` for search results: evidence by score descending, then path,
/// then line; `past_reviews` follow the evidence they annotate and are ordered
/// by `pr_number, path, line, created_at` (the store query's ORDER BY, so they
/// never depend on insertion order).
pub const SEARCH_ORDERING: &str = "score_desc,path_asc,line_asc";
/// `pack.ordering` for connection expansions.
pub const EXPANSION_ORDERING: &str = "confidence_desc,path_asc";

/// Estimated token count of `text`: `ceil(utf8_bytes / 3)` (`"bytes3-v1"`).
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(BYTES_PER_TOKEN)
}

/// The exact text a tool result is sent to the client as. The MCP layer uses
/// this same function, so what is measured is what is returned.
pub fn render(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_default()
}

/// Estimated tokens of a tool result as returned to the client.
pub fn measure(value: &Value) -> usize {
    estimate_tokens(&render(value))
}

/// Total order for evidence packets: score descending, then path ascending,
/// then start line ascending, then symbol, then id. No two distinct packets
/// compare equal, so the order never depends on input order or hash iteration.
pub fn compare_packets(a: &EvidencePacket, b: &EvidencePacket) -> Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.path.cmp(&b.path))
        .then_with(|| a.start_line.unwrap_or(0).cmp(&b.start_line.unwrap_or(0)))
        .then_with(|| a.symbol.cmp(&b.symbol))
        .then_with(|| a.id.cmp(&b.id))
}

/// One array of units in a tool result.
pub struct SectionSpec {
    /// Key of the array in the result object.
    pub key: &'static str,
    /// What the units are, reported as `kind` on omitted entries.
    pub kind: &'static str,
    /// Builds the reduced form of a unit that does not fit whole: identity and
    /// score only, plus an `expansion_handle` to fetch the rest. `None` when
    /// the unit has no useful reduced form. The second argument is the anchor
    /// paths of the result (the paths in its `likely_files` section, in order;
    /// empty when it has none), for handles that need a source file.
    pub minimal: fn(&Value, &[String]) -> Option<Value>,
    /// Index of an earlier section; a unit here is only eligible while a unit
    /// with the same path from that section is included (reviews follow the
    /// evidence they annotate).
    pub requires_path_from: Option<usize>,
}

/// A candidate that an existing result cap (not the budget) withheld.
pub struct CapOmission {
    pub path: String,
    pub symbol: Option<String>,
    pub kind: &'static str,
    /// `"below_cap"` or `"recommendation_cap"`.
    pub reason: &'static str,
}

/// How to pack one tool result.
pub struct PackPlan {
    /// `None` keeps every unit (the default: existing caps only).
    pub budget: Option<usize>,
    pub ordering: &'static str,
    /// Priority order: earlier sections are filled first.
    pub sections: Vec<SectionSpec>,
    /// Candidates the existing caps withheld, listed in `pack.omitted`.
    pub cap_omissions: Vec<CapOmission>,
    /// Further cap-withheld candidates that are counted but not enumerated.
    pub cap_unlisted: usize,
}

fn evidence_minimal(v: &Value, _anchors: &[String]) -> Option<Value> {
    let path = v.get("path")?.as_str()?;
    let mut out = Map::new();
    for key in [
        "id",
        "type",
        "path",
        "symbol",
        "start_line",
        "end_line",
        "score",
    ] {
        if let Some(x) = v.get(key) {
            out.insert(key.to_owned(), x.clone());
        }
    }
    out.insert("minimal".to_owned(), json!(true));
    let mut handle = Map::new();
    handle.insert("path".to_owned(), json!(path));
    if let Some(symbol) = v.get("symbol") {
        handle.insert("symbol".to_owned(), symbol.clone());
    }
    out.insert("expansion_handle".to_owned(), Value::Object(handle));
    Some(Value::Object(out))
}

/// Reduced form of an expansion entry. Its handle must be callable as is:
/// `explain_connection` needs a `source` and a different `target`, so the
/// source is the first anchor that is not the entry itself. An entry that is
/// the only anchor has nothing to be explained against, so its handle is
/// `expand_connections` on that path instead.
fn scored_path_minimal(v: &Value, anchors: &[String]) -> Option<Value> {
    let path = v.get("path")?.as_str()?;
    let handle = match anchors.iter().find(|a| a.as_str() != path) {
        Some(source) => json!({
            "tool": "explain_connection",
            "source": source,
            "target": path,
        }),
        None => json!({ "tool": "expand_connections", "paths": [path] }),
    };
    Some(json!({
        "path": path,
        "confidence": v.get("confidence").cloned().unwrap_or(Value::Null),
        "minimal": true,
        "expansion_handle": handle,
    }))
}

fn review_minimal(v: &Value, _anchors: &[String]) -> Option<Value> {
    let path = v.get("path")?.as_str()?;
    let mut out = Map::new();
    for key in ["pr_number", "path", "line", "pr_merged"] {
        if let Some(x) = v.get(key) {
            out.insert(key.to_owned(), x.clone());
        }
    }
    out.insert("minimal".to_owned(), json!(true));
    out.insert(
        "expansion_handle".to_owned(),
        json!({ "tool": "get_review_history", "path": path }),
    );
    Some(Value::Object(out))
}

fn no_minimal(_: &Value, _: &[String]) -> Option<Value> {
    None
}

/// Plan for `search_context`: ranked evidence, then the reviewer comments on
/// the evidence that was kept. `cap_dropped` are packets the search's own cap
/// cut (in rank order); `beyond_first_cap` further fused candidates it cut
/// before building packets.
///
/// Note for readers of `pack.truncated` / `pack.omitted_count` on a search:
/// `beyond_first_cap` counts every fused candidate past `top_k`, not only
/// evidence that would otherwise have been shown, so `truncated: true` is
/// common even with no budget. It means "more candidates existed than the
/// result caps return", not "the budget cut something".
pub fn search_plan(
    budget: Option<usize>,
    cap_dropped: &[EvidencePacket],
    beyond_first_cap: usize,
) -> PackPlan {
    PackPlan {
        budget,
        ordering: SEARCH_ORDERING,
        sections: vec![
            SectionSpec {
                key: "evidence",
                kind: "evidence",
                minimal: evidence_minimal,
                requires_path_from: None,
            },
            SectionSpec {
                key: "past_reviews",
                kind: "review_comment",
                minimal: review_minimal,
                requires_path_from: Some(0),
            },
        ],
        cap_omissions: cap_dropped
            .iter()
            .map(|p| CapOmission {
                path: p.path.clone(),
                symbol: p.symbol.clone(),
                kind: "evidence",
                reason: "below_cap",
            })
            .collect(),
        cap_unlisted: beyond_first_cap,
    }
}

/// Plan for `expand_connections`. `dropped` are the candidates the per-anchor
/// and per-kind recommendation caps withheld, as `(kind, path)`.
pub fn expansion_plan(budget: Option<usize>, dropped: &[(&'static str, String)]) -> PackPlan {
    PackPlan {
        budget,
        ordering: EXPANSION_ORDERING,
        sections: vec![
            SectionSpec {
                key: "likely_files",
                kind: "likely_file",
                minimal: scored_path_minimal,
                requires_path_from: None,
            },
            SectionSpec {
                key: "cochange_expansions",
                kind: "cochange",
                minimal: scored_path_minimal,
                requires_path_from: None,
            },
            SectionSpec {
                key: "import_expansions",
                kind: "import",
                minimal: scored_path_minimal,
                requires_path_from: None,
            },
            SectionSpec {
                key: "likely_tests",
                kind: "test",
                minimal: no_minimal,
                requires_path_from: None,
            },
        ],
        cap_omissions: dropped
            .iter()
            .map(|(kind, path)| CapOmission {
                path: path.clone(),
                symbol: None,
                kind,
                reason: "recommendation_cap",
            })
            .collect(),
        cap_unlisted: 0,
    }
}

struct Unit {
    path: String,
    symbol: Option<String>,
    full: Value,
    minimal: Option<Value>,
}

/// Indentation of an item of a section array in the rendered result.
const ITEM_INDENT: usize = 4;
/// Indentation of an entry of `pack.omitted` in the rendered result.
const OMITTED_INDENT: usize = 6;
/// Bytes of slack for the small things that change when a unit moves between
/// omitted and kept: brackets and separators of an array going from empty to
/// non-empty, the digits of `omitted_count`, `true`/`false`.
const SIZE_SLACK: usize = 32;

/// Rendered-size bounds of one JSON value placed inside the result.
struct FormCost {
    /// Never more than its rendered size: the compact serialization.
    lo: usize,
    /// Never less than the bytes it adds as a section item (pretty-printed,
    /// indented, with separators).
    hi: usize,
    pretty_len: usize,
    lines: usize,
}

impl FormCost {
    fn of(v: &Value) -> Self {
        let pretty = render(v);
        let lines = pretty.matches('\n').count() + 1;
        let mut cost = FormCost {
            lo: serde_json::to_string(v).map_or(0, |s| s.len()),
            hi: 0,
            pretty_len: pretty.len(),
            lines,
        };
        cost.hi = cost.nested_hi(ITEM_INDENT);
        cost
    }

    /// Upper bound on the bytes this value occupies when nested at `indent`.
    fn nested_hi(&self, indent: usize) -> usize {
        self.pretty_len + indent * self.lines + SIZE_SLACK
    }
}

/// What the greedy fill knows about a unit without composing the result.
/// Moving a unit from omitted to kept adds at least `lo` of its form and
/// removes at most `saved` (its entry in `pack.omitted`), and adds at most `hi`
/// plus the longest omitted entry another unit's omission could be replaced by.
struct UnitCost {
    section: usize,
    unit: usize,
    full: FormCost,
    minimal: Option<FormCost>,
    saved: usize,
    /// The least that this unit or any later one can add to the size
    /// (cheapest form minus `saved`); negative when keeping can shrink it.
    floor: i128,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    Omitted,
    Full,
    Minimal,
}

/// Fields that survive the degraded layouts: they identify the snapshot the
/// answer (or its absence) refers to. The pack-only layout drops all of them,
/// so `Packer::pack` mirrors the mismatch/staleness/warnings as flags.
const ESSENTIAL_KEYS: [&str; 4] = [
    "snapshot",
    "snapshot_mismatch",
    "index_warnings",
    "index_state",
];

/// How much of the result is kept around the pack: `rung` 0 keeps every base
/// field and the section arrays; 1 keeps only the snapshot fields; 2 is the
/// pack alone. Rungs above 0 hold no units. `list_limit` bounds `pack.omitted`.
#[derive(Clone, Copy)]
struct Layout {
    rung: usize,
    list_limit: usize,
}

struct Packer {
    base: Map<String, Value>,
    plan: PackPlan,
    sections: Vec<Vec<Unit>>,
    forms: Vec<Vec<Form>>,
}

impl Packer {
    fn included_paths(&self, section: usize) -> HashSet<&str> {
        self.sections[section]
            .iter()
            .zip(&self.forms[section])
            .filter(|(_, form)| **form != Form::Omitted)
            .map(|(unit, _)| unit.path.as_str())
            .collect()
    }

    /// The whole result for the current selection.
    fn compose(&self, layout: Layout, used: usize, within_budget: bool) -> Value {
        let mut out: Map<String, Value> = match layout.rung {
            0 => self.base.clone(),
            1 => self
                .base
                .iter()
                .filter(|(k, _)| ESSENTIAL_KEYS.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            _ => Map::new(),
        };
        if layout.rung == 0 {
            for (spec, (units, forms)) in self
                .plan
                .sections
                .iter()
                .zip(self.sections.iter().zip(self.forms.iter()))
            {
                let items: Vec<Value> = units
                    .iter()
                    .zip(forms)
                    .filter_map(|(unit, form)| match form {
                        Form::Full => Some(unit.full.clone()),
                        Form::Minimal => unit.minimal.clone(),
                        Form::Omitted => None,
                    })
                    .collect();
                out.insert(spec.key.to_owned(), Value::Array(items));
            }
        }
        out.insert(
            "pack".to_owned(),
            self.pack(layout.list_limit, used, within_budget),
        );
        Value::Object(out)
    }

    fn pack(&self, list_limit: usize, used: usize, within_budget: bool) -> Value {
        let mut omitted: Vec<Value> = Vec::new();
        let mut count = 0usize;
        let mut any_minimal = false;
        for (spec, (units, forms)) in self
            .plan
            .sections
            .iter()
            .zip(self.sections.iter().zip(self.forms.iter()))
        {
            for (unit, form) in units.iter().zip(forms) {
                match form {
                    Form::Minimal => any_minimal = true,
                    Form::Omitted => {
                        count += 1;
                        if omitted.len() < list_limit {
                            omitted.push(omitted_entry(
                                &unit.path,
                                unit.symbol.as_deref(),
                                spec.kind,
                                "over_budget",
                            ));
                        }
                    }
                    Form::Full => {}
                }
            }
        }
        for cap in &self.plan.cap_omissions {
            count += 1;
            if omitted.len() < list_limit {
                omitted.push(omitted_entry(
                    &cap.path,
                    cap.symbol.as_deref(),
                    cap.kind,
                    cap.reason,
                ));
            }
        }
        count += self.plan.cap_unlisted;
        let mut pack = json!({
            "estimator": TOKEN_ESTIMATOR,
            "budget_tokens": self.plan.budget,
            "used_tokens": used,
            "truncated": count > 0 || any_minimal,
            "omitted_count": count,
            "omitted": omitted,
            "ordering": self.plan.ordering,
            "within_budget": within_budget,
        });
        // Tiny flags that mirror the snapshot fields the degraded layouts drop,
        // so a `within_budget: true` pack never hides a snapshot mismatch or a
        // possibly stale index. Present only when they apply; counted in the
        // measurement like everything else in the pack.
        if let Some(obj) = pack.as_object_mut() {
            for (flag, applies) in self.disclosed_flags() {
                if applies {
                    obj.insert(flag.to_owned(), json!(true));
                }
            }
        }
        pack
    }

    /// `(pack key, applies)` for the flags that survive every layout rung.
    fn disclosed_flags(&self) -> [(&'static str, bool); 3] {
        let stale = self
            .base
            .get("index_state")
            .and_then(|s| s.get("stale"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let has_warnings = self
            .base
            .get("index_warnings")
            .and_then(Value::as_array)
            .is_some_and(|w| !w.is_empty());
        [
            (
                "snapshot_mismatch",
                self.base.contains_key("snapshot_mismatch"),
            ),
            ("stale", stale),
            ("has_warnings", has_warnings),
        ]
    }

    /// Whether the selection fits. Placeholders are the worst case for their
    /// own size (`used_tokens` as wide as the budget, `within_budget` false),
    /// so the real, final values can only be shorter.
    fn fits(&self, layout: Layout, budget: usize) -> bool {
        measure(&self.compose(layout, budget, false)) <= budget
    }

    /// Exact rendered size in bytes of the selection, with the same
    /// placeholders as [`Packer::fits`].
    fn rendered_len(&self, layout: Layout, budget: usize) -> usize {
        render(&self.compose(layout, budget, false)).len()
    }

    fn set_all(&mut self, form: Form) {
        for forms in &mut self.forms {
            forms.fill(form);
        }
    }

    fn run(mut self) -> Value {
        let Some(budget) = self.plan.budget else {
            self.set_all(Form::Full);
            let layout = Layout {
                rung: 0,
                list_limit: MAX_OMITTED_LISTED,
            };
            return self.finish(layout, None);
        };
        self.set_all(Form::Omitted);
        let limit = (budget / BUDGET_TOKENS_PER_LISTED_OMISSION).min(MAX_OMITTED_LISTED);
        // Richest layout that holds the pack with every unit still omitted.
        // Each step down gives up something before the budget is exceeded.
        let ladder = [
            Layout {
                rung: 0,
                list_limit: limit,
            },
            Layout {
                rung: 1,
                list_limit: limit,
            },
            Layout {
                rung: 2,
                list_limit: limit,
            },
            Layout {
                rung: 2,
                list_limit: limit / 2,
            },
            Layout {
                rung: 2,
                list_limit: 0,
            },
        ];
        let fallback = Layout {
            rung: 2,
            list_limit: 0,
        };
        let mut layout = ladder
            .into_iter()
            .find(|l| self.fits(*l, budget))
            .unwrap_or(fallback);
        if layout.rung == 0 {
            self.fill_greedily(layout, budget);
            self.trim_until_fits(layout, budget);
            // Nothing at all fit: answer with the pack alone (plus the fields
            // that say which snapshot and how fresh), not a shell of empty
            // arrays that reads like "no results".
            let has_units = self.sections.iter().any(|units| !units.is_empty());
            let kept_any = self.forms.iter().flatten().any(|f| *f != Form::Omitted);
            if has_units && !kept_any {
                layout = ladder
                    .into_iter()
                    .filter(|l| l.rung > 0)
                    .find(|l| self.fits(*l, budget))
                    .unwrap_or(fallback);
            }
        }
        self.finish(layout, Some(budget))
    }

    /// Walk the units in priority order. A unit goes in whole if the whole
    /// result still fits; otherwise in its reduced form if that fits;
    /// otherwise it is skipped, and the walk goes on, since a later unit may
    /// be smaller.
    ///
    /// Measuring a candidate by composing and rendering the whole result is
    /// O(result), so doing it for every unit is O(n^2). Instead the walk keeps
    /// `[lo, hi]` bounds on the byte size of the current selection and only
    /// composes when a candidate cannot be decided from the bounds (see
    /// [`UnitCost`]). The bounds are conservative in both directions, so every
    /// decision equals the one an exact measurement would have made; once no
    /// remaining unit can possibly fit, the walk stops early.
    fn fill_greedily(&mut self, layout: Layout, budget: usize) {
        let limit = budget.saturating_mul(BYTES_PER_TOKEN);
        let costs = self.unit_costs();
        let pull_in = self.largest_omitted_entry();
        let mut lo = self.rendered_len(layout, budget);
        let mut hi = lo;
        for cost in &costs {
            if lo as i128 + cost.floor > limit as i128 {
                break;
            }
            let (s, u) = (cost.section, cost.unit);
            if let Some(parent) = self.plan.sections[s].requires_path_from {
                if !self
                    .included_paths(parent)
                    .contains(self.sections[s][u].path.as_str())
                {
                    continue;
                }
            }
            let mut candidates = vec![(Form::Full, &cost.full)];
            if let Some(minimal) = &cost.minimal {
                candidates.push((Form::Minimal, minimal));
            }
            for (form, form_cost) in candidates {
                let lo_after = (lo + form_cost.lo).saturating_sub(cost.saved);
                if lo_after > limit {
                    continue; // cannot fit, whatever the pack looks like
                }
                self.forms[s][u] = form;
                let hi_after = hi + form_cost.hi + pull_in;
                if hi_after <= limit {
                    // Certainly fits.
                    (lo, hi) = (lo_after, hi_after);
                    break;
                }
                let exact = self.rendered_len(layout, budget);
                if exact <= limit {
                    (lo, hi) = (exact, exact);
                    break;
                }
                self.forms[s][u] = Form::Omitted;
            }
        }
    }

    /// Size bounds for every unit in priority order. See [`UnitCost`].
    fn unit_costs(&self) -> Vec<UnitCost> {
        let mut costs = Vec::new();
        for (s, units) in self.sections.iter().enumerate() {
            for (u, unit) in units.iter().enumerate() {
                let entry = omitted_entry(
                    &unit.path,
                    unit.symbol.as_deref(),
                    self.plan.sections[s].kind,
                    "over_budget",
                );
                let full = FormCost::of(&unit.full);
                let minimal = unit.minimal.as_ref().map(FormCost::of);
                let cheapest = minimal.as_ref().map_or(full.lo, |m| m.lo.min(full.lo));
                let saved = FormCost::of(&entry).nested_hi(OMITTED_INDENT);
                costs.push(UnitCost {
                    section: s,
                    unit: u,
                    full,
                    minimal,
                    saved,
                    floor: cheapest as i128 - saved as i128,
                });
            }
        }
        // Suffix minimum: the least any unit from here on can add to the size.
        let mut running = i128::MAX;
        for cost in costs.iter_mut().rev() {
            running = running.min(cost.floor);
            cost.floor = running;
        }
        costs
    }

    /// Upper bound on the size of the longest entry that could be listed in
    /// `pack.omitted`. Omitting one fewer unit can list a different entry in
    /// its place, which may be longer than the one it replaces.
    fn largest_omitted_entry(&self) -> usize {
        let from_units = self.sections.iter().enumerate().flat_map(|(s, units)| {
            let kind = self.plan.sections[s].kind;
            units.iter().map(move |unit| {
                let entry = omitted_entry(&unit.path, unit.symbol.as_deref(), kind, "over_budget");
                FormCost::of(&entry).nested_hi(OMITTED_INDENT)
            })
        });
        let from_caps = self.plan.cap_omissions.iter().map(|cap| {
            let entry = omitted_entry(&cap.path, cap.symbol.as_deref(), cap.kind, cap.reason);
            FormCost::of(&entry).nested_hi(OMITTED_INDENT)
        });
        from_units.chain(from_caps).max().unwrap_or(0)
    }

    /// Safety net: the greedy pass measures every step, so this is expected to
    /// do nothing, but the contract is the final measurement, so check it.
    fn trim_until_fits(&mut self, layout: Layout, budget: usize) {
        while !self.fits(layout, budget) {
            let last = (0..self.sections.len()).rev().find_map(|s| {
                (0..self.sections[s].len())
                    .rev()
                    .find(|&u| self.forms[s][u] != Form::Omitted)
                    .map(|u| (s, u))
            });
            match last {
                Some((s, u)) => self.forms[s][u] = Form::Omitted,
                None => break,
            }
        }
    }

    fn finish(self, layout: Layout, budget: Option<usize>) -> Value {
        // `used_tokens` is part of the text it measures, so find the least
        // value that is at least the measured size of the result carrying it.
        let within = |used: usize| match budget {
            Some(b) => measure(&self.compose(layout, used, true)) <= b,
            None => true,
        };
        let mut used = 0usize;
        loop {
            let measured = measure(&self.compose(layout, used, within(used)));
            if measured <= used {
                break;
            }
            used = measured;
        }
        let within_budget = within(used);
        self.compose(layout, used, within_budget)
    }
}

fn omitted_entry(path: &str, symbol: Option<&str>, kind: &str, reason: &str) -> Value {
    let mut entry = Map::new();
    entry.insert("path".to_owned(), json!(path));
    if let Some(symbol) = symbol {
        entry.insert("symbol".to_owned(), json!(symbol));
    }
    // Evidence is the default unit; other kinds are named so that a path that
    // is cut as both a co-change and an import candidate reads unambiguously.
    if kind != "evidence" {
        entry.insert("kind".to_owned(), json!(kind));
    }
    entry.insert("reason".to_owned(), json!(reason));
    Value::Object(entry)
}

fn unit_path(item: &Value) -> String {
    match item {
        Value::String(s) => s.clone(),
        other => other
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
    }
}

/// Turn a finished tool result into a pack (see the module docs). `result`
/// must be the complete object that will be sent to the client, including
/// decoration such as the snapshot id: the budget covers all of it. A result
/// that is not an object is returned unchanged.
///
/// With a budget the returned value serializes (via [`render`]) to at most
/// `budget` estimated tokens, unless even a bare `pack` object cannot fit, in
/// which case it is returned alone with `pack.within_budget` false rather than
/// pretending. Without a budget every unit is kept and the pack only reports.
///
/// The pack carries tiny flags that survive every layout rung, present only
/// when they apply: `snapshot_mismatch: true` (the result had a
/// `snapshot_mismatch`), `stale: true` (`index_state.stale`) and
/// `has_warnings: true` (non-empty `index_warnings`). They are counted in the
/// measurement, so a pack-only answer still cannot hide a mismatch or
/// staleness.
///
/// Determinism: for a given index, query, budget and clock the output is
/// byte-identical, independent of insertion order (every source of ordering has
/// a total tie-break; see `pack.ordering`). The clock matters: search ranking
/// adds a `recency_boost` computed from the current time, so byte-identity
/// across calls holds only while that clock is fixed (or the boost is
/// negligible); it is not a guarantee across days.
pub fn apply(result: Value, plan: PackPlan) -> Value {
    let Value::Object(base) = result else {
        return result;
    };
    new_packer(base, plan).run()
}

/// Split the sections off `base` into units, everything omitted to start with.
fn new_packer(mut base: Map<String, Value>, plan: PackPlan) -> Packer {
    // Anchor files (the `likely_files` of an expansion), for handles that need
    // a source file. Taken before the sections are split off the result.
    let anchors: Vec<String> = base
        .get("likely_files")
        .and_then(Value::as_array)
        .map(|items| items.iter().map(unit_path).collect())
        .unwrap_or_default();
    let mut sections = Vec::new();
    for spec in &plan.sections {
        let items = match base.remove(spec.key) {
            Some(Value::Array(items)) => items,
            _ => Vec::new(),
        };
        sections.push(
            items
                .into_iter()
                .map(|item| Unit {
                    path: unit_path(&item),
                    symbol: item
                        .get("symbol")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    minimal: (spec.minimal)(&item, &anchors),
                    full: item,
                })
                .collect::<Vec<_>>(),
        );
    }
    let forms = sections
        .iter()
        .map(|units| vec![Form::Omitted; units.len()])
        .collect();
    Packer {
        base,
        plan,
        sections,
        forms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(path: &str, symbol: Option<&str>, line: usize, score: f32) -> EvidencePacket {
        EvidencePacket {
            id: format!("ev_{path}_{line}"),
            kind: engram_domain::EvidenceKind::Symbol,
            title: format!("{path}:{line}"),
            path: path.to_owned(),
            symbol: symbol.map(str::to_owned),
            snippet: None,
            start_line: Some(line),
            end_line: Some(line + 3),
            score,
            signals: vec!["bm25".to_owned()],
        }
    }

    #[test]
    fn estimator_is_ceil_bytes_over_three_and_monotonic() {
        assert_eq!(TOKEN_ESTIMATOR, "bytes3-v1");
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1);
        assert_eq!(estimate_tokens("abc"), 1);
        assert_eq!(estimate_tokens("abcd"), 2);
        // Bytes, not chars: a 3-byte char is one token's worth.
        assert_eq!(estimate_tokens("\u{20ac}"), 1);
        let mut previous = 0;
        for n in 0..200 {
            let t = estimate_tokens(&"x".repeat(n));
            assert!(t >= previous);
            previous = t;
        }
    }

    #[test]
    fn packet_order_is_total_and_ignores_input_order() {
        let packets = vec![
            packet("b.rs", Some("f"), 10, 1.0),
            packet("a.rs", Some("g"), 20, 1.0),
            packet("a.rs", Some("f"), 5, 1.0),
            packet("a.rs", Some("a"), 5, 1.0),
            packet("c.rs", None, 0, 2.0),
            packet("a.rs", Some("a"), 5, 0.5),
        ];
        let mut expected = packets.clone();
        expected.sort_by(compare_packets);
        let ids: Vec<&str> = expected.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids[0], "ev_c.rs_0", "highest score first");
        for rotation in 0..packets.len() {
            let mut shuffled = packets.clone();
            shuffled.rotate_left(rotation);
            shuffled.reverse();
            shuffled.sort_by(compare_packets);
            let got: Vec<&str> = shuffled.iter().map(|p| p.id.as_str()).collect();
            assert_eq!(got, ids, "rotation {rotation}");
        }
    }

    fn many_units(n: usize) -> Value {
        let evidence: Vec<Value> = (0..n)
            .map(|i| {
                json!({
                    "id": format!("ev_{i:03}"),
                    "type": "symbol",
                    "title": format!("handler_{i} in src/m{i}.rs:{}", i * 10 + 1),
                    "path": format!("src/m{i}.rs"),
                    "symbol": format!("handler_{i}"),
                    "snippet": format!("fn handler_{i}() {{\n{}\n}}", "    work();\n".repeat(30)),
                    "start_line": i * 10 + 1,
                    "end_line": i * 10 + 32,
                    "score": 1.0 - (i as f64) / 100.0,
                    "signals": ["bm25"],
                })
            })
            .collect();
        json!({ "task": "t", "snapshot": "abc", "evidence": evidence, "past_reviews": [] })
    }

    #[test]
    fn a_budget_bounds_the_rendered_result_and_the_pack_discloses_the_rest() {
        let plan = |b| search_plan(b, &[], 0);
        let unbounded = apply(many_units(12), plan(None));
        assert_eq!(unbounded["evidence"].as_array().unwrap().len(), 12);
        assert_eq!(unbounded["pack"]["omitted_count"], 0);
        assert_eq!(unbounded["pack"]["budget_tokens"], Value::Null);
        assert_eq!(unbounded["pack"]["truncated"], false);

        let packed = apply(many_units(12), plan(Some(900)));
        assert!(measure(&packed) <= 900, "{}", measure(&packed));
        let pack = &packed["pack"];
        assert_eq!(pack["truncated"], true);
        assert_eq!(pack["estimator"], "bytes3-v1");
        assert_eq!(pack["within_budget"], true);
        let kept = packed["evidence"].as_array().unwrap().len();
        assert!(kept > 0 && kept < 12, "kept {kept}");
        let omitted = pack["omitted"].as_array().unwrap();
        assert_eq!(pack["omitted_count"].as_u64().unwrap() as usize, 12 - kept);
        assert!(omitted.iter().all(|o| o["reason"] == "over_budget"));
        assert!(omitted.iter().all(|o| o["path"].is_string()));
        // used_tokens is an upper bound on the measured size and tight to it.
        let used = pack["used_tokens"].as_u64().unwrap() as usize;
        assert!(used >= measure(&packed) && used <= measure(&packed) + 1);
    }

    #[test]
    fn the_omitted_list_is_capped_but_counts_everything() {
        let packed = apply(many_units(60), search_plan(Some(6000), &[], 0));
        assert!(measure(&packed) <= 6000);
        let pack = &packed["pack"];
        assert_eq!(
            pack["omitted"].as_array().unwrap().len(),
            MAX_OMITTED_LISTED
        );
        let kept = packed["evidence"].as_array().unwrap().len();
        assert_eq!(pack["omitted_count"].as_u64().unwrap() as usize, 60 - kept);
    }

    #[test]
    fn a_unit_too_big_whole_is_returned_as_a_minimal_record_with_a_handle() {
        let one = apply(many_units(1), search_plan(Some(300), &[], 0));
        assert!(measure(&one) <= 300, "{}", measure(&one));
        let evidence = one["evidence"].as_array().unwrap();
        assert_eq!(evidence.len(), 1);
        let record = &evidence[0];
        assert_eq!(record["minimal"], true);
        assert!(record.get("snippet").is_none());
        assert_eq!(record["path"], "src/m0.rs");
        assert_eq!(record["symbol"], "handler_0");
        assert_eq!(record["start_line"], 1);
        assert_eq!(record["end_line"], 32);
        assert!(record["score"].is_number());
        assert_eq!(record["expansion_handle"]["path"], "src/m0.rs");
        assert_eq!(record["expansion_handle"]["symbol"], "handler_0");
        assert_eq!(one["pack"]["truncated"], true);
        assert_eq!(one["pack"]["omitted_count"], 0);
    }

    #[test]
    fn when_nothing_fits_only_the_pack_is_returned() {
        let tiny = apply(many_units(5), search_plan(Some(150), &[], 0));
        let keys: Vec<&String> = tiny.as_object().unwrap().keys().collect();
        assert!(
            keys.iter().all(|k| *k == "pack" || *k == "snapshot"),
            "{keys:?}"
        );
        assert!(tiny.get("evidence").is_none());
        assert_eq!(tiny["pack"]["omitted_count"], 5);
        assert_eq!(tiny["pack"]["truncated"], true);

        // Below the size of a bare pack the budget cannot be met; say so.
        let impossible = apply(many_units(5), search_plan(Some(5), &[], 0));
        assert_eq!(impossible.as_object().unwrap().len(), 1);
        assert_eq!(impossible["pack"]["within_budget"], false);
        assert_eq!(impossible["pack"]["omitted_count"], 5);
    }

    #[test]
    fn cap_withheld_candidates_are_disclosed_even_without_a_budget() {
        let dropped = vec![
            ("cochange", "a.rs".to_owned()),
            ("import", "b.rs".to_owned()),
        ];
        let result = json!({
            "likely_files": [], "likely_tests": [],
            "cochange_expansions": [], "import_expansions": [],
        });
        let packed = apply(result, expansion_plan(None, &dropped));
        let pack = &packed["pack"];
        assert_eq!(pack["omitted_count"], 2);
        assert_eq!(pack["truncated"], true);
        assert_eq!(pack["omitted"][0]["reason"], "recommendation_cap");
        assert_eq!(pack["omitted"][0]["kind"], "cochange");
        assert_eq!(pack["omitted"][1]["path"], "b.rs");
    }

    #[test]
    fn reviews_are_only_kept_for_evidence_that_was_kept() {
        let mut result = many_units(3);
        result["past_reviews"] = json!([
            { "pr_number": 1, "path": "src/m0.rs", "body": "x".repeat(40), "pr_merged": true },
            { "pr_number": 2, "path": "src/m2.rs", "body": "y".repeat(40), "pr_merged": true },
        ]);
        let all = apply(result.clone(), search_plan(None, &[], 0));
        assert_eq!(all["past_reviews"].as_array().unwrap().len(), 2);
        // Room for the first evidence unit and its review, not for the rest.
        let evidence_only = measure(&apply(many_units(1), search_plan(None, &[], 0)));
        let packed = apply(result, search_plan(Some(evidence_only + 150), &[], 0));
        assert!(measure(&packed) <= evidence_only + 150);
        let kept: Vec<&str> = packed["evidence"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["path"].as_str().unwrap())
            .collect();
        for review in packed["past_reviews"].as_array().unwrap() {
            assert!(kept.contains(&review["path"].as_str().unwrap()), "{review}");
        }
    }

    #[test]
    fn packing_is_deterministic() {
        let a = render(&apply(many_units(12), search_plan(Some(900), &[], 0)));
        let b = render(&apply(many_units(12), search_plan(Some(900), &[], 0)));
        assert_eq!(a, b);
    }

    fn flagged_result() -> Value {
        let mut result = many_units(5);
        result["snapshot_mismatch"] = json!({ "requested": "old", "current": "abc" });
        result["index_warnings"] = json!(["the file watcher stopped"]);
        result["index_state"] = json!({ "status": "ready", "generation": 3, "stale": true });
        result
    }

    #[test]
    fn mismatch_staleness_and_warnings_survive_every_layout_rung() {
        let mut layouts = HashSet::new();
        for budget in 1..=1500 {
            let packed = apply(flagged_result(), search_plan(Some(budget), &[], 0));
            let pack = &packed["pack"];
            if pack["within_budget"] != true {
                continue;
            }
            assert!(measure(&packed) <= budget, "budget {budget}");
            for flag in ["snapshot_mismatch", "stale", "has_warnings"] {
                assert_eq!(
                    pack[flag], true,
                    "budget {budget}: {flag} missing in {pack}"
                );
            }
            let layout = if packed.get("evidence").is_some() {
                "full"
            } else if packed.get("snapshot").is_some() {
                "snapshot_fields"
            } else {
                assert_eq!(packed.as_object().unwrap().len(), 1, "pack only");
                "pack_only"
            };
            layouts.insert(layout);
        }
        assert_eq!(
            layouts,
            HashSet::from(["full", "snapshot_fields", "pack_only"]),
            "the sweep should visit every rung that can be within budget"
        );

        // No budget: the flags are reported next to the untouched fields.
        let unbounded = apply(flagged_result(), search_plan(None, &[], 0));
        assert_eq!(unbounded["pack"]["snapshot_mismatch"], true);
        assert_eq!(unbounded["pack"]["stale"], true);
        assert_eq!(unbounded["pack"]["has_warnings"], true);
        assert!(unbounded.get("snapshot_mismatch").is_some());

        // Nothing to report: no flags at all, so a healthy pack is unchanged.
        for result in [many_units(5), {
            let mut r = many_units(5);
            r["index_warnings"] = json!([]);
            r["index_state"] = json!({ "status": "ready", "stale": false });
            r
        }] {
            let pack = apply(result, search_plan(Some(900), &[], 0))["pack"].clone();
            for flag in ["snapshot_mismatch", "stale", "has_warnings"] {
                assert!(pack.get(flag).is_none(), "{flag} in {pack}");
            }
        }
    }

    #[test]
    fn a_minimal_expansion_handle_names_a_source_so_it_is_callable() {
        let anchors = vec!["app/a.py".to_owned(), "app/b.py".to_owned()];
        let entry = |path: &str| json!({ "path": path, "confidence": 0.5, "reason": "r" });
        let handle = |path: &str, anchors: &[String]| {
            scored_path_minimal(&entry(path), anchors).unwrap()["expansion_handle"].clone()
        };
        assert_eq!(
            handle("app/c.py", &anchors),
            json!({ "tool": "explain_connection", "source": "app/a.py", "target": "app/c.py" })
        );
        // The first anchor is not its own source: the next one is used.
        assert_eq!(handle("app/a.py", &anchors)["source"], "app/b.py");
        // A lone anchor has nothing to be explained against.
        assert_eq!(
            handle("app/a.py", &anchors[..1]),
            json!({ "tool": "expand_connections", "paths": ["app/a.py"] })
        );

        // Through the real packer: a squeezed expansion entry gets such a handle.
        let reason = "x".repeat(600);
        let result = json!({
            "likely_files": [{ "path": "app/a.py", "confidence": 1.0, "reason": "anchor" }],
            "likely_tests": [],
            "cochange_expansions": [
                { "path": "app/b.py", "confidence": 0.9, "reason": reason }
            ],
            "import_expansions": [],
        });
        let mut seen = false;
        for budget in 100..1500 {
            let packed = apply(result.clone(), expansion_plan(Some(budget), &[]));
            for entry in packed["cochange_expansions"]
                .as_array()
                .into_iter()
                .flatten()
            {
                if entry["minimal"] == true {
                    seen = true;
                    let handle = &entry["expansion_handle"];
                    assert_eq!(handle["tool"], "explain_connection");
                    assert_eq!(handle["source"], "app/a.py");
                    assert_eq!(handle["target"], "app/b.py");
                }
            }
        }
        assert!(seen, "some budget should have produced a minimal record");
    }

    fn anchors_result(n: usize) -> Value {
        let likely: Vec<Value> = (0..n)
            .map(|i| {
                json!({
                    "path": format!("src/anchor_{i:05}.rs"),
                    "confidence": 1.0,
                    "reason": "given as an anchor file",
                })
            })
            .collect();
        json!({
            "likely_files": likely,
            "likely_tests": [],
            "cochange_expansions": [],
            "import_expansions": [],
        })
    }

    #[test]
    fn thousands_of_anchors_are_packed_without_quadratic_work() {
        // Sanity, not a benchmark: this used to re-render the whole result for
        // every unit. The assertions that matter are about the output; the
        // generous time limit only guards against a hang.
        let started = std::time::Instant::now();
        let small = apply(anchors_result(2000), expansion_plan(Some(900), &[]));
        let large = apply(anchors_result(2000), expansion_plan(Some(10_000_000), &[]));
        assert!(started.elapsed() < std::time::Duration::from_secs(60));

        assert!(measure(&small) <= 900, "{}", measure(&small));
        assert_eq!(small["pack"]["within_budget"], true);
        let kept = small["likely_files"].as_array().unwrap().len();
        assert!(kept > 0 && kept < 2000, "kept {kept}");
        assert_eq!(
            small["pack"]["omitted_count"].as_u64().unwrap() as usize + kept,
            2000
        );

        assert_eq!(large["likely_files"].as_array().unwrap().len(), 2000);
        assert_eq!(large["pack"]["omitted_count"], 0);
        assert_eq!(large["pack"]["truncated"], false);
        assert!(measure(&large) <= 10_000_000);
    }

    /// The greedy fill as it was before the size bounds: measure every
    /// candidate by composing and rendering the whole result.
    fn reference_fill(packer: &mut Packer, layout: Layout, budget: usize) {
        for s in 0..packer.sections.len() {
            for u in 0..packer.sections[s].len() {
                if let Some(parent) = packer.plan.sections[s].requires_path_from {
                    if !packer
                        .included_paths(parent)
                        .contains(packer.sections[s][u].path.as_str())
                    {
                        continue;
                    }
                }
                packer.forms[s][u] = Form::Full;
                if packer.fits(layout, budget) {
                    continue;
                }
                if packer.sections[s][u].minimal.is_some() {
                    packer.forms[s][u] = Form::Minimal;
                    if packer.fits(layout, budget) {
                        continue;
                    }
                }
                packer.forms[s][u] = Form::Omitted;
            }
        }
    }

    #[test]
    fn the_bounded_fill_decides_exactly_like_measuring_every_candidate() {
        let mut search = many_units(14);
        search["past_reviews"] = json!([
            { "pr_number": 1, "path": "src/m0.rs", "body": "x".repeat(300), "pr_merged": true },
            { "pr_number": 2, "path": "src/m3.rs", "body": "y".repeat(40), "pr_merged": true },
            { "pr_number": 3, "path": "src/m9.rs", "body": "z".repeat(900), "pr_merged": false },
        ]);
        let scored = |prefix: &str, n: usize| -> Vec<Value> {
            (0..n)
                .map(|i| {
                    json!({
                        "path": format!("{prefix}/file_{i:02}_{}.py", "n".repeat(i % 7)),
                        "confidence": 1.0 - (i as f64) / 100.0,
                        "reason": "r".repeat(20 + (i * 53) % 400),
                    })
                })
                .collect()
        };
        let expansion = json!({
            "likely_files": scored("app", 12),
            "cochange_expansions": scored("co", 9),
            "import_expansions": scored("imp", 9),
            "likely_tests": (0..6).map(|i| format!("tests/test_{i}.py")).collect::<Vec<_>>(),
        });
        let dropped = vec![
            (
                "cochange",
                "co/a_dropped_candidate_with_a_long_path.py".to_owned(),
            ),
            ("import", "imp/short.py".to_owned()),
        ];
        type PlanFor = Box<dyn Fn(usize) -> PackPlan>;
        let cases: Vec<(&str, Value, PlanFor)> = vec![
            ("search", search, Box::new(|b| search_plan(Some(b), &[], 3))),
            (
                "expansion",
                expansion,
                Box::new(move |b| expansion_plan(Some(b), &dropped)),
            ),
        ];
        for (name, result, plan) in &cases {
            let Value::Object(base) = result.clone() else {
                unreachable!()
            };
            let mut checked_partial = false;
            for budget in (150..=9000).step_by(37) {
                let layout = Layout {
                    rung: 0,
                    list_limit: (budget / BUDGET_TOKENS_PER_LISTED_OMISSION)
                        .min(MAX_OMITTED_LISTED),
                };
                let mut fast = new_packer(base.clone(), plan(budget));
                let mut slow = new_packer(base.clone(), plan(budget));
                fast.set_all(Form::Omitted);
                slow.set_all(Form::Omitted);
                fast.fill_greedily(layout, budget);
                reference_fill(&mut slow, layout, budget);
                assert!(
                    fast.forms == slow.forms,
                    "{name}, budget {budget}: bounded fill differs from the exact one"
                );
                let kept = fast
                    .forms
                    .iter()
                    .flatten()
                    .filter(|f| **f != Form::Omitted)
                    .count();
                let all = fast.forms.iter().flatten().count();
                checked_partial |= kept > 0 && kept < all;
            }
            assert!(checked_partial, "{name}: no budget produced a partial pack");
        }
    }
}
