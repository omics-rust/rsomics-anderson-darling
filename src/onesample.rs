//! One-sample Anderson-Darling goodness-of-fit, `scipy.stats.anderson`.
//!
//! A² = −n − Σᵢ (2i−1)/n · [ln F(z₍ᵢ₎) + ln(1−F(z₍ₙ₊₁₋ᵢ₎))], where the data are
//! standardized by the per-distribution fit. The critical values and
//! significance levels come straight from scipy's tabulated arrays; the
//! interpolated p-value is `np.interp(A², critical, sig/100)`.

use serde::Serialize;

use rsomics_common::{Result, RsomicsError};

use crate::dist::{Dist, fit_and_logcdf_logsf};

#[derive(Debug, Clone, Serialize)]
pub struct AndersonResult {
    pub dist: String,
    pub statistic: f64,
    pub critical_values: Vec<f64>,
    pub significance_level: Vec<f64>,
    /// Interpolated p-value, `np.interp(A², critical_values, sig/100)`.
    pub pvalue: f64,
    pub fit_params: Vec<f64>,
}

pub fn anderson(data: &[f64], dist: Dist) -> Result<AndersonResult> {
    if data.len() < 2 {
        return Err(RsomicsError::InvalidInput(
            "anderson needs at least two observations".into(),
        ));
    }
    if data.iter().any(|v| v.is_nan()) {
        return Err(RsomicsError::InvalidInput("input contains NaN".into()));
    }
    let n = data.len();
    let fit = fit_and_logcdf_logsf(dist, data)?;

    // A² = -n - Σ (2i-1)/n · (logcdf[i] + logsf[n+1-i]); logsf reversed.
    let nf = n as f64;
    let terms: Vec<f64> = (0..n)
        .map(|i| {
            let coef = (2.0 * (i as f64 + 1.0) - 1.0) / nf;
            coef * (fit.logcdf[i] + fit.logsf[n - 1 - i])
        })
        .collect();
    let a2 = -nf - crate::sum::pairwise_sum(&terms);

    let critical = dist.critical_values(n);
    let sig = dist.significance_levels().to_vec();
    let sig_frac: Vec<f64> = sig.iter().map(|s| s / 100.0).collect();
    let pvalue = interp(a2, &critical, &sig_frac);

    Ok(AndersonResult {
        dist: dist.as_str().to_string(),
        statistic: a2,
        critical_values: critical,
        significance_level: sig,
        pvalue,
        fit_params: fit.params,
    })
}

/// `np.interp(x, xp, fp)`: piecewise-linear with flat extrapolation at both
/// ends. `xp` is ascending (scipy's `critical_values`).
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    let mut hi = 1;
    while x > xp[hi] {
        hi += 1;
    }
    let lo = hi - 1;
    let t = (x - xp[lo]) / (xp[hi] - xp[lo]);
    fp[lo] + t * (fp[hi] - fp[lo])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_statistic_matches_scipy() {
        // anderson(x, 'norm') statistic, scipy 1.17.1.
        let x = [
            0.5, 1.2, -0.3, 2.1, 0.8, -1.1, 1.5, 0.2, -0.7, 1.9, 0.4, -0.5, 1.1, 0.9, -0.2,
        ];
        let r = anderson(&x, Dist::Norm).unwrap();
        assert!(
            (r.statistic - 0.1340984312286544).abs() < 1e-12,
            "{}",
            r.statistic
        );
        assert_eq!(r.critical_values.len(), 5);
        assert_eq!(r.significance_level, vec![15.0, 10.0, 5.0, 2.5, 1.0]);
    }

    #[test]
    fn interp_flat_extrapolation() {
        let xp = [1.0, 2.0, 3.0];
        let fp = [0.1, 0.05, 0.01];
        assert!((interp(0.5, &xp, &fp) - 0.1).abs() < 1e-15);
        assert!((interp(5.0, &xp, &fp) - 0.01).abs() < 1e-15);
        assert!((interp(1.5, &xp, &fp) - 0.075).abs() < 1e-15);
    }

    #[test]
    fn expon_below_support_is_positive_infinity() {
        // scipy 1.17.1: anderson([-1,2,3], 'expon') == inf (the -1 is below the
        // [0, ∞) support, so logcdf=-inf pushes A² to +inf).
        let r = anderson(&[-1.0, 2.0, 3.0], Dist::Expon).unwrap();
        assert!(
            r.statistic.is_infinite() && r.statistic > 0.0,
            "{}",
            r.statistic
        );
    }

    #[test]
    fn rejects_short_input() {
        assert!(anderson(&[1.0], Dist::Norm).is_err());
    }

    #[test]
    fn rejects_nan() {
        assert!(anderson(&[1.0, f64::NAN, 2.0], Dist::Norm).is_err());
    }
}
