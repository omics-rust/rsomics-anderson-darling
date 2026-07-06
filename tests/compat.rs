//! Value-exact compatibility against `scipy.stats.anderson` / `anderson_ksamp`.
//!
//! Expected values were computed once with scipy 1.17.1 (`tests/golden/`) and
//! frozen in `expected.json`; no scipy runs at test time. The A² statistic goes
//! through the cephes `ndtr`/`log_ndtr` port plus the distribution MLE fits and
//! must match scipy to ~1e-9; critical values match the tables exactly; the
//! k-sample interpolated p-value matches to ~1e-10.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use rsomics_anderson_darling::{Dist, anderson, anderson_ksamp, parse_values};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    one_sample: Vec<OneCase>,
    one_sample_boundary: Vec<BoundaryCase>,
    one_sample_degenerate: Vec<BoundaryCase>,
    k_sample: Vec<KCase>,
}

#[derive(Deserialize)]
struct BoundaryCase {
    name: String,
    dist: String,
    /// scipy statistic encoded as a string so `nan`/`inf` (JSON has no native
    /// non-finite literals) survive the round-trip; a plain decimal string is a
    /// finite defined value.
    statistic: String,
}

#[derive(Deserialize)]
struct OneCase {
    name: String,
    dist: String,
    statistic: f64,
    pvalue: f64,
    critical: Vec<f64>,
    sig: Vec<f64>,
}

#[derive(Deserialize)]
struct KCase {
    name: String,
    nfiles: usize,
    midrank: bool,
    statistic: f64,
    pvalue: f64,
    critical: Vec<f64>,
}

fn golden_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn load(path: &Path) -> Vec<f64> {
    let f = File::open(path).unwrap_or_else(|_| panic!("open {}", path.display()));
    parse_values(BufReader::new(f)).expect("parse")
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(f64::MIN_POSITIVE)
}

fn read_golden() -> Golden {
    let f = File::open(golden_dir().join("expected.json")).expect("open expected.json");
    serde_json::from_reader(BufReader::new(f)).expect("parse expected.json")
}

#[test]
fn one_sample_matches_scipy() {
    let g = read_golden();
    for c in &g.one_sample {
        let data = load(&golden_dir().join(format!("{}.tsv", c.name)));
        let dist = Dist::parse(&c.dist).unwrap();
        let r = anderson(&data, dist).unwrap();

        // norm/expon/gumbel_r/gumbel_l fit analytically or via a root solved to
        // full precision, so the statistic is bit-exact (last-bit). logistic
        // standardizes by scipy's `fsolve` iterate (a path-specific, non-fully-
        // converged trust-region point); we converge the same MLE to the true
        // root, which agrees with scipy only to ~1e-6 — a documented HELD
        // boundary, not a bug. The p-value is exact regardless.
        let tol = if c.dist == "logistic" { 1e-5 } else { 1e-12 };
        let sr = rel(r.statistic, c.statistic);
        assert!(
            sr <= tol,
            "{}: A² {} vs scipy {} (rel {sr:e})",
            c.name,
            r.statistic,
            c.statistic
        );
        assert_eq!(
            r.critical_values, c.critical,
            "{}: critical values mismatch",
            c.name
        );
        assert_eq!(
            r.significance_level, c.sig,
            "{}: significance levels mismatch",
            c.name
        );
        let pr = rel(r.pvalue, c.pvalue);
        assert!(
            pr <= 1e-10,
            "{}: p {} vs scipy {} (rel {pr:e})",
            c.name,
            r.pvalue,
            c.pvalue
        );
    }
}

#[test]
#[ignore = "diagnostic: prints worst relative error vs frozen scipy goldens"]
fn report_worst_relerr() {
    let g = read_golden();
    let mut worst_stat = 0.0_f64;
    let mut worst_p = 0.0_f64;
    let mut worst_one = 0.0_f64;
    for c in &g.one_sample {
        let data = load(&golden_dir().join(format!("{}.tsv", c.name)));
        let r = anderson(&data, Dist::parse(&c.dist).unwrap()).unwrap();
        let e = rel(r.statistic, c.statistic);
        eprintln!("  one {:14} stat rel-err {e:e}", c.name);
        worst_one = worst_one.max(e);
        worst_stat = worst_stat.max(e);
        worst_p = worst_p.max(rel(r.pvalue, c.pvalue));
    }
    eprintln!("WORST one-sample statistic rel-err = {worst_one:e}");
    for c in &g.k_sample {
        let samples: Vec<Vec<f64>> = (1..=c.nfiles)
            .map(|i| load(&golden_dir().join(format!("{}_s{i}.tsv", c.name))))
            .collect();
        let r = anderson_ksamp(&samples, c.midrank).unwrap();
        let e = rel(r.statistic, c.statistic);
        eprintln!("  k   {:14} stat rel-err {e:e}", c.name);
        worst_stat = worst_stat.max(e);
        worst_p = worst_p.max(rel(r.pvalue, c.pvalue));
    }
    eprintln!("WORST statistic rel-err = {worst_stat:e}");
    eprintln!("WORST p-value   rel-err = {worst_p:e}");
}

#[test]
fn one_sample_boundary_matches_scipy() {
    // A data value outside the fitted distribution's support drives the
    // logcdf/logsf term to -∞/0 and the A² statistic to +∞, exactly as scipy
    // (`expon.anderson` on data below the [0, ∞) support). We used to emit NaN.
    let g = read_golden();
    for c in &g.one_sample_boundary {
        let data = load(&golden_dir().join(format!("{}.tsv", c.name)));
        let dist = Dist::parse(&c.dist).unwrap();
        let r = anderson(&data, dist).unwrap();
        assert_eq!(c.statistic, "inf", "{}: unexpected golden encoding", c.name);
        assert!(
            r.statistic.is_infinite() && r.statistic > 0.0,
            "{}: A² {} vs scipy +inf",
            c.name,
            r.statistic
        );
    }
}

#[test]
fn single_observation_fails_loud() {
    // scipy returns nan for a 1-element sample; we refuse it (a defined test
    // needs ≥2 observations) rather than emit a spurious finite A². Fail-loud:
    // non-zero exit with a stderr message.
    use std::io::Write;
    use std::process::Command;
    let mut child = Command::new(env!("CARGO_BIN_EXE_rsomics-anderson-darling"))
        .args(["--dist", "norm", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"5\n").unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(!out.status.success(), "expected non-zero exit on n=1");
    assert!(
        !out.stderr.is_empty(),
        "expected a stderr message on n=1 failure"
    );
}

#[test]
fn one_sample_degenerate_matches_scipy() {
    // Zero-variance (constant) data. scipy's per-dist defined values (scipy
    // 1.17.1): norm→nan, expon→1.376…, logistic→nan, gumbel_r/gumbel_l→+∞.
    // The gumbel scale MLE underflows to the smallest normal with loc→∞, so A²
    // diverges to +∞ instead of the spurious finite value a naive scale root
    // would give.
    let g = read_golden();
    for c in &g.one_sample_degenerate {
        let data = load(&golden_dir().join(format!("{}.tsv", c.name)));
        let dist = Dist::parse(&c.dist).unwrap();
        let got = anderson(&data, dist).unwrap().statistic;
        match c.statistic.as_str() {
            "nan" => assert!(got.is_nan(), "{} {}: A² {got} vs scipy nan", c.name, c.dist),
            "inf" => assert!(
                got.is_infinite() && got > 0.0,
                "{} {}: A² {got} vs scipy +inf",
                c.name,
                c.dist
            ),
            s => {
                let want: f64 = s.parse().unwrap();
                assert!(
                    rel(got, want) <= 1e-12,
                    "{} {}: A² {got} vs scipy {want}",
                    c.name,
                    c.dist
                );
            }
        }
    }
}

#[test]
fn k_sample_matches_scipy() {
    let g = read_golden();
    for c in &g.k_sample {
        let samples: Vec<Vec<f64>> = (1..=c.nfiles)
            .map(|i| load(&golden_dir().join(format!("{}_s{i}.tsv", c.name))))
            .collect();
        let r = anderson_ksamp(&samples, c.midrank).unwrap();

        let sr = rel(r.statistic, c.statistic);
        assert!(
            sr <= 1e-9,
            "{}: A²kN {} vs scipy {} (rel {sr:e})",
            c.name,
            r.statistic,
            c.statistic
        );
        for (got, want) in r.critical_values.iter().zip(c.critical.iter()) {
            assert!(
                rel(*got, *want) <= 1e-9,
                "{}: critical {got} vs scipy {want}",
                c.name
            );
        }
        let pr = rel(r.pvalue, c.pvalue);
        assert!(
            pr <= 1e-10,
            "{}: p {} vs scipy {} (rel {pr:e})",
            c.name,
            r.pvalue,
            c.pvalue
        );
    }
}
