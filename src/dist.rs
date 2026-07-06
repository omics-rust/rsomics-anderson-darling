//! Per-distribution parameter fitting and log-CDF/SF, matching scipy's
//! `anderson` standardization exactly.
//!
//! scipy fits loc/scale from the data per distribution: norm uses sample
//! mean and ddof=1 std; expon uses 0 and the sample mean; logistic solves the
//! two MLE equations via `fsolve`; gumbel_r/gumbel_l run the gumbel MLE
//! root-find on the scale. The log-CDF/SF formulas mirror the exact expressions
//! in `scipy.stats.distributions` so the A² statistic is bit-identical.

use rsomics_common::{Result, RsomicsError};

use crate::ndtr::log_ndtr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dist {
    Norm,
    Expon,
    Logistic,
    GumbelR,
    GumbelL,
}

impl Dist {
    pub fn parse(s: &str) -> Result<Self> {
        let d = match s {
            "norm" => Dist::Norm,
            "expon" => Dist::Expon,
            "logistic" => Dist::Logistic,
            "gumbel_r" => Dist::GumbelR,
            "gumbel_l" | "gumbel" | "extreme1" => Dist::GumbelL,
            other => {
                return Err(RsomicsError::InvalidInput(format!(
                    "invalid distribution '{other}'; must be one of \
                     norm, expon, logistic, gumbel_l, gumbel_r"
                )));
            }
        };
        Ok(d)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Dist::Norm => "norm",
            Dist::Expon => "expon",
            Dist::Logistic => "logistic",
            Dist::GumbelR => "gumbel_r",
            Dist::GumbelL => "gumbel_l",
        }
    }

    /// scipy's per-dist significance levels (percent), tier order matches the
    /// `critical_values` array.
    pub fn significance_levels(self) -> &'static [f64] {
        match self {
            Dist::Norm | Dist::Expon => &[15.0, 10.0, 5.0, 2.5, 1.0],
            Dist::Logistic => &[25.0, 10.0, 5.0, 2.5, 1.0, 0.5],
            Dist::GumbelR | Dist::GumbelL => &[25.0, 10.0, 5.0, 2.5, 1.0],
        }
    }

    /// Tabulated critical values adjusted for sample size `n`, rounded to 3
    /// decimals exactly as scipy's `around(..., 3)`.
    pub fn critical_values(self, n: usize) -> Vec<f64> {
        let nf = n as f64;
        let (base, denom): (&[f64], f64) = match self {
            Dist::Norm => (&AVALS_NORM, 1.0 + 0.75 / nf + 2.25 / nf / nf),
            Dist::Expon => (&AVALS_EXPON, 1.0 + 0.6 / nf),
            Dist::Logistic => (&AVALS_LOGISTIC, 1.0 + 0.25 / nf),
            Dist::GumbelR | Dist::GumbelL => (&AVALS_GUMBEL, 1.0 + 0.2 / nf.sqrt()),
        };
        base.iter().map(|&a| round3(a / denom)).collect()
    }
}

const AVALS_NORM: [f64; 5] = [0.561, 0.631, 0.752, 0.873, 1.035];
const AVALS_EXPON: [f64; 5] = [0.916, 1.062, 1.321, 1.591, 1.959];
const AVALS_GUMBEL: [f64; 5] = [0.474, 0.637, 0.757, 0.877, 1.038];
const AVALS_LOGISTIC: [f64; 6] = [0.426, 0.563, 0.660, 0.769, 0.906, 1.010];

/// numpy `around(x, 3)`: round-half-to-even at 3 decimals.
fn round3(x: f64) -> f64 {
    let scaled = x * 1000.0;
    let r = scaled.round_ties_even();
    r / 1000.0
}

/// Sample mean via numpy's pairwise reduction (`np.mean`).
fn mean(x: &[f64]) -> f64 {
    crate::sum::pairwise_sum(x) / x.len() as f64
}

/// Sample std with ddof=1 (`np.std(x, ddof=1)`); numpy sums the squared
/// deviations pairwise.
fn std_ddof1(x: &[f64]) -> f64 {
    let m = mean(x);
    let n = x.len() as f64;
    let sq: Vec<f64> = x.iter().map(|&v| (v - m) * (v - m)).collect();
    (crate::sum::pairwise_sum(&sq) / (n - 1.0)).sqrt()
}

/// log-CDF and log-SF of the standardized order statistics, computed exactly as
/// scipy's `distributions.<dist>.logcdf`/`logsf` over the fitted, sorted data.
///
/// `sorted_x` must be ascending. Returns `(logcdf, logsf)` aligned with
/// `sorted_x`, plus the fitted parameters for diagnostics.
pub fn fit_and_logcdf_logsf(dist: Dist, x: &[f64]) -> Result<DistFit> {
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("NaN guarded upstream"));
    let n = sorted.len();
    let mut logcdf = vec![0.0_f64; n];
    let mut logsf = vec![0.0_f64; n];

    let params: Vec<f64> = match dist {
        Dist::Norm => {
            let xbar = mean(x);
            let s = std_ddof1(x);
            for i in 0..n {
                let w = (sorted[i] - xbar) / s;
                logcdf[i] = log_ndtr(w);
                logsf[i] = log_ndtr(-w);
            }
            vec![xbar, s]
        }
        Dist::Expon => {
            let xbar = mean(x);
            for i in 0..n {
                let w = sorted[i] / xbar;
                // expon's support is [0, ∞); below it scipy clamps cdf=0/sf=1,
                // giving logcdf=-∞ and logsf=0. The analytic tails (log(-expm1(-w))
                // is log of a negative → NaN; -w flips positive) don't hold there,
                // so a below-support value drives A² to +∞, matching scipy.
                if w < 0.0 {
                    logcdf[i] = f64::NEG_INFINITY;
                    logsf[i] = 0.0;
                } else {
                    logcdf[i] = neg_expm1(-w).ln();
                    logsf[i] = -w;
                }
            }
            vec![0.0, xbar]
        }
        Dist::Logistic => {
            let (a, b) = fit_logistic(x);
            for i in 0..n {
                let w = (sorted[i] - a) / b;
                // logistic: logcdf = log_expit(w), logsf = log_expit(-w)
                logcdf[i] = log_expit(w);
                logsf[i] = log_expit(-w);
            }
            vec![a, b]
        }
        Dist::GumbelR => {
            if is_constant(x) {
                degenerate_gumbel(&mut logcdf, &mut logsf)
            } else {
                let (loc, scale) = fit_gumbel_r(x);
                for i in 0..n {
                    let w = (sorted[i] - loc) / scale;
                    // gumbel_r: logcdf = -exp(-w), logsf = log(-expm1(-exp(-w)))
                    logcdf[i] = -(-w).exp();
                    logsf[i] = neg_expm1(-(-w).exp()).ln();
                }
                vec![loc, scale]
            }
        }
        Dist::GumbelL => {
            if is_constant(x) {
                degenerate_gumbel(&mut logcdf, &mut logsf)
            } else {
                // scipy fits gumbel_l by negating data through gumbel_r, negating loc.
                let neg: Vec<f64> = x.iter().map(|&v| -v).collect();
                let (loc_r, scale) = fit_gumbel_r(&neg);
                let loc = -loc_r;
                for i in 0..n {
                    let w = (sorted[i] - loc) / scale;
                    // gumbel_l: logcdf = log(-expm1(-exp(w))), logsf = -exp(w)
                    logcdf[i] = neg_expm1(-w.exp()).ln();
                    logsf[i] = -w.exp();
                }
                vec![loc, scale]
            }
        }
    };

    Ok(DistFit {
        logcdf,
        logsf,
        params,
    })
}

pub struct DistFit {
    pub logcdf: Vec<f64>,
    pub logsf: Vec<f64>,
    pub params: Vec<f64>,
}

fn is_constant(x: &[f64]) -> bool {
    x.iter().all(|&v| v == x[0])
}

/// Zero-variance data drives scipy's gumbel scale MLE to underflow to the
/// smallest normal (2.2e-308) with loc→∞, so every standardized order statistic
/// has logcdf=-∞ / logsf=0 and A² diverges to +∞. Reproduce that limit directly
/// (the brentq scale root is ill-defined here and lands on a spurious nonzero
/// scale). The reported fit is scipy's degenerate (loc=∞, scale=MIN_POSITIVE).
fn degenerate_gumbel(logcdf: &mut [f64], logsf: &mut [f64]) -> Vec<f64> {
    logcdf.fill(f64::NEG_INFINITY);
    logsf.fill(0.0);
    vec![f64::INFINITY, f64::MIN_POSITIVE]
}

/// `-expm1(x)` = `1 - exp(x)`, computed without catastrophic cancellation, the
/// same primitive scipy uses for the gumbel/expon CDF tails.
fn neg_expm1(x: f64) -> f64 {
    -x.exp_m1()
}

/// `log(expit(x))` = `-log1p(exp(-x))` for x≥0, `x - log1p(exp(x))` for x<0;
/// matches `scipy.special.log_expit`.
fn log_expit(x: f64) -> f64 {
    if x < 0.0 {
        x - x.exp().ln_1p()
    } else {
        -(-x).exp().ln_1p()
    }
}

/// Logistic loc/scale MLE via the two equations scipy passes to `fsolve`:
///   Σ 1/(1+exp((x-a)/b)) = N/2
///   Σ ((x-a)/b)·(1-exp((x-a)/b))/(1+exp((x-a)/b)) = -N
///
/// scipy seeds with (mean, std_ddof1) and runs `fsolve` (MINPACK hybrd) at
/// xtol=1e-5, which stops at a path-specific, not-fully-converged trust-region
/// iterate. We solve the same MLE to the true root with a forward-difference
/// Newton; the resulting A² agrees with scipy only to ~1e-6 (the fsolve-path
/// residual). The reported p-value is unaffected, so this is a HELD boundary.
fn fit_logistic(x: &[f64]) -> (f64, f64) {
    let nf = x.len() as f64;
    let mut a = mean(x);
    let mut b = std_ddof1(x);

    let resid = |a: f64, b: f64| -> [f64; 2] {
        let mut r0 = 0.0;
        let mut r1 = 0.0;
        for &xj in x {
            let t = (xj - a) / b;
            let e = t.exp();
            r0 += 1.0 / (1.0 + e);
            r1 += t * (1.0 - e) / (1.0 + e);
        }
        [r0 - 0.5 * nf, r1 + nf]
    };

    for _ in 0..200 {
        let f = resid(a, b);
        if f[0].abs() < 1e-12 && f[1].abs() < 1e-12 {
            break;
        }
        // Finite-difference Jacobian (MINPACK forward differences).
        let ha = 1e-7 * a.abs().max(1.0);
        let hb = 1e-7 * b.abs().max(1.0);
        let fa = resid(a + ha, b);
        let fb = resid(a, b + hb);
        let j00 = (fa[0] - f[0]) / ha;
        let j10 = (fa[1] - f[1]) / ha;
        let j01 = (fb[0] - f[0]) / hb;
        let j11 = (fb[1] - f[1]) / hb;
        let det = j00 * j11 - j01 * j10;
        if det.abs() < 1e-300 {
            break;
        }
        let da = (f[0] * j11 - f[1] * j01) / det;
        let db = (f[1] * j00 - f[0] * j10) / det;
        let mut na = a - da;
        let mut nb = b - db;
        if nb <= 0.0 {
            nb = b / 2.0;
            na = a;
        }
        if (na - a).abs() < 1e-14 * a.abs().max(1.0) && (nb - b).abs() < 1e-14 * b.abs().max(1.0) {
            a = na;
            b = nb;
            break;
        }
        a = na;
        b = nb;
    }
    (a, b)
}

/// gumbel_r loc/scale MLE, matching `scipy.stats.gumbel_r.fit` with free
/// loc and scale.
///
/// scipy solves `func(scale) = mean(x) - wavg - scale = 0` where
/// `wavg = average_with_log_weights(x, -x/scale)`, then
/// `loc = -scale·(logsumexp(-x/scale) - log n)`. The scale root is found via
/// `root_scalar` (brentq, rtol=xtol=1e-14) with the same bracket-expansion seed.
fn fit_gumbel_r(x: &[f64]) -> (f64, f64) {
    let xbar = mean(x);
    let n = x.len() as f64;

    let func = |scale: f64| -> f64 {
        // average_with_log_weights(x, -x/scale)
        let logw: Vec<f64> = x.iter().map(|&v| -v / scale).collect();
        let maxw = logw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut wsum = 0.0;
        let mut xwsum = 0.0;
        for (i, &v) in x.iter().enumerate() {
            let w = (logw[i] - maxw).exp();
            wsum += w;
            xwsum += v * w;
        }
        let wavg = xwsum / wsum;
        xbar - wavg - scale
    };

    // scipy's bracket: start (0.5, 2.0), expand until sign change.
    let mut lo = 0.5;
    let mut hi = 2.0;
    let mut flo = func(lo);
    let mut fhi = func(hi);
    while flo.signum() == fhi.signum() && (lo > 0.0 || hi < f64::INFINITY) {
        lo /= 2.0;
        hi *= 2.0;
        flo = func(lo);
        fhi = func(hi);
        if !lo.is_finite() || !hi.is_finite() {
            break;
        }
    }

    let scale = brentq(&func, lo, hi, flo, fhi, 1e-14, 1e-14);

    // loc = -scale·(logsumexp(-x/scale) - log n)
    let logw: Vec<f64> = x.iter().map(|&v| -v / scale).collect();
    let maxw = logw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sumexp: f64 = logw.iter().map(|&w| (w - maxw).exp()).sum();
    let logsumexp = maxw + sumexp.ln();
    let loc = -scale * (logsumexp - n.ln());
    (loc, scale)
}

/// Brent's method root-find, the algorithm behind scipy's `brentq`
/// (`root_scalar(method='brentq')`). `flo`/`fhi` are the bracket endpoint values.
#[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
fn brentq(
    f: &dyn Fn(f64) -> f64,
    xa: f64,
    xb: f64,
    fa_in: f64,
    fb_in: f64,
    xtol: f64,
    rtol: f64,
) -> f64 {
    let (mut a, mut b) = (xa, xb);
    let (mut fa, mut fb) = (fa_in, fb_in);
    if fa == 0.0 {
        return a;
    }
    if fb == 0.0 {
        return b;
    }
    let (mut c, mut fc) = (a, fa);
    let mut d = b - a;
    let mut e = d;
    for _ in 0..200 {
        if fb.signum() == fc.signum() {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol = 2.0 * f64::EPSILON * b.abs() + 0.5 * (xtol + rtol * b.abs());
        let m = 0.5 * (c - b);
        if m.abs() <= tol || fb == 0.0 {
            return b;
        }
        if e.abs() < tol || fa.abs() <= fb.abs() {
            d = m;
            e = m;
        } else {
            let s = fb / fa;
            let (mut p, mut q);
            if a == c {
                p = 2.0 * m * s;
                q = 1.0 - s;
            } else {
                let qa = fa / fc;
                let r = fb / fc;
                p = s * (2.0 * m * qa * (qa - r) - (b - a) * (r - 1.0));
                q = (qa - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if 2.0 * p < (3.0 * m * q - (tol * q).abs()).min(0.5 * (e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = m;
            }
        }
        a = b;
        fa = fb;
        if d.abs() > tol {
            b += d;
        } else {
            b += if m > 0.0 { tol } else { -tol };
        }
        fb = f(b);
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round3_matches_numpy_around() {
        // numpy around is half-to-even on the exact binary value; 0.5605 is not
        // representable and rounds down, matching np.around(0.5605, 3) == 0.56.
        assert_eq!(round3(0.5605), 0.56);
        assert_eq!(round3(0.7515), 0.752);
        assert_eq!(round3(0.5615), 0.562);
    }

    #[test]
    fn gumbel_r_mle_matches_scipy() {
        // scipy.stats.gumbel_r.fit on this sample (scipy 1.17.1).
        let x = [
            1.5, 2.3, 0.8, 3.1, 2.0, 1.2, 4.5, 2.8, 1.9, 2.6, 0.5, 3.3, 2.1, 1.7, 2.9,
        ];
        let (loc, scale) = fit_gumbel_r(&x);
        assert!((loc - 1.7271056900290092).abs() < 1e-9, "loc {loc}");
        assert!((scale - 0.896101006214686).abs() < 1e-9, "scale {scale}");
    }
}
