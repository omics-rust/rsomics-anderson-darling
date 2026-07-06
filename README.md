# rsomics-anderson-darling

Anderson-Darling goodness-of-fit tests, value-exact to SciPy 1.17.1:

- **One-sample** (`scipy.stats.anderson`): the A² statistic for a fitted
  `norm`, `expon`, `logistic`, `gumbel_l`, or `gumbel_r` distribution, with
  SciPy's tabulated critical values, significance levels, and the interpolated
  p-value.
- **k-sample** (`scipy.stats.anderson_ksamp`): the Scholz-Stephens A²kN
  statistic (midrank or right variant) with the normalized statistic and the
  interpolated/clipped p-value.

## Install

```
cargo install rsomics-anderson-darling
```

## Usage

One-sample — a single column of values per line plus `--dist`:

```
rsomics-anderson-darling data.tsv --dist norm
```

```
# dist	norm
# statistic	0.15325780479418682
# pvalue	0.15
significance	critical
15	0.55
10	0.619
5	0.737
2.5	0.856
1	1.015
```

`--dist` accepts `norm` (default), `expon`, `logistic`, `gumbel_l`, `gumbel_r`.
`-` or an omitted path reads stdin.

k-sample — two or more value files with `--ksamp`, output `statistic<TAB>p`:

```
rsomics-anderson-darling s1.tsv s2.tsv s3.tsv --ksamp
```

```
8.246950644536524	0.001
```

The midrank variant (SciPy's default, ties handled) is used unless `--right` is
passed. `--json` emits a single machine-readable result envelope.

## Value-exactness

| dist | A² statistic vs SciPy 1.17.1 |
|---|---|
| `norm` | bit-exact (pairwise sum matches `np.sum`/`np.mean`/`np.std`) |
| `expon` | bit-exact |
| `gumbel_r` / `gumbel_l` | bit-exact (gumbel MLE solved to full precision); see zero-variance boundary below |
| `logistic` | ~1e-6 — see boundary below |
| k-sample (midrank + right) | bit-exact / ≤1 ULP |

Critical values match SciPy's tables exactly; the interpolated and capped/floored
p-values are bit-exact.

**Logistic boundary.** SciPy fits the logistic loc/scale with `optimize.fsolve`
(MINPACK `hybrd`) at `xtol=1e-5`, which stops at a path-specific, not-fully-
converged trust-region iterate; the reported A² depends on that exact iterate.
This crate solves the same maximum-likelihood equations to the true root, which
agrees with SciPy's reported statistic only to ~1e-6. The reported p-value is
unaffected (logistic critical values are tabulated and the test statistic is
typically below the cap), so this is a held compatibility boundary rather than a
defect.

**Zero-variance (constant) data.** On a constant sample SciPy's per-distribution
defined values are: `norm`→`nan`, `expon`→`1.376…` (finite), `logistic`→`nan`,
`gumbel_r`/`gumbel_l`→`+inf`. This crate matches all five. For the gumbel fits
SciPy's scale MLE underflows to the smallest normal (`2.2e-308`) with `loc→∞`, so
A² diverges to `+inf`; we reproduce that limit directly rather than let the scale
root-find land on a spurious finite value. A one-observation sample is refused
(non-zero exit) instead of returning SciPy's `nan` — a defined test needs at least
two observations. A *near*-constant sample (two distinct values very close) is a
separate held boundary: `brentq` conditioning can drift the statistic by ~1.6e-5,
which is not corrected.

## Performance

Single-thread, Apple M2, SciPy 1.17.1 / NumPy 2.4.6 with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`, ours forced to one
thread with `-t1`:

| test | fixture | ours `-t1` full | SciPy full | full ratio | compute-only ratio |
|---|---|---|---|---|---|
| one-sample `norm` | N = 2,000,000 | 336 ms | 1.439 s | 4.29× | 1.82× |
| k-sample (3 samples) | N = 2,000,000 | 931 ms | 2.113 s | 2.27× | 1.08× |

The one-sample win is driven by a faster parser and a tighter standardization
loop; the k-sample compute-only margin is thin because SciPy runs the
`searchsorted` inner loops in vectorized C, but our faster I/O still wins the
end-to-end pipeline.

## Origin

This crate is an independent Rust reimplementation of `scipy.stats.anderson` and
`scipy.stats.anderson_ksamp` based on:

- The published methods:
  - T. W. Anderson and D. A. Darling, "Asymptotic Theory of Certain Goodness of
    Fit Criteria Based on Stochastic Processes", Ann. Math. Statist. 23 (1952).
  - F. W. Scholz and M. A. Stephens, "K-Sample Anderson-Darling Tests", JASA 82
    (1987), 918-924.
  - M. A. Stephens, "Goodness of Fit for the Extreme Value Distribution",
    Biometrika 64 (1977); and "Tests of Fit for the Logistic Distribution",
    Biometrika 66 (1979).
- The SciPy 1.17.1 source for the tabulated critical-value arrays, the
  significance-level tiers, and the b0/b1/b2 interpolation
  (`scipy/stats/_morestats.py`, BSD-3-Clause).

The normal CDF / log-CDF (`norm.logcdf`, `norm.logsf`) is a port of Cephes
`ndtr` plus the SciPy `xsf` `log_ndtr` region split (Moshier, public domain),
which makes the `norm` A² statistic bit-identical to `scipy.special.log_ndtr`.
The numeric reduction order matches NumPy's pairwise `np.sum`.

Golden test values were generated once with SciPy 1.17.1 and frozen; the
compatibility tests run without SciPy.

License: MIT OR Apache-2.0.
Upstream credit: SciPy (https://scipy.org, BSD-3-Clause); Cephes (Stephen L.
Moshier, public domain).
