//! k-sample Anderson-Darling test, `scipy.stats.anderson_ksamp` (Scholz &
//! Stephens 1987).
//!
//! Computes the A²kN statistic (midrank, eq. 7, or right, eq. 6), normalizes by
//! `(A²kN − (k−1))/√σ²`, and interpolates the p-value from the tabulated
//! critical values via the probit-of-significance quadratic fit — capped at 0.25
//! and floored at 0.001 exactly as scipy 1.17.

use serde::Serialize;

use rsomics_common::{Result, RsomicsError};

#[derive(Debug, Clone, Serialize)]
pub struct KSampResult {
    pub statistic: f64,
    pub critical_values: Vec<f64>,
    pub significance_level: Vec<f64>,
    pub pvalue: f64,
    /// True when the p-value hit scipy's 0.25 cap or 0.001 floor.
    pub p_clipped: bool,
}

pub fn anderson_ksamp(samples: &[Vec<f64>], midrank: bool) -> Result<KSampResult> {
    let k = samples.len();
    if k < 2 {
        return Err(RsomicsError::InvalidInput(
            "anderson_ksamp needs at least two samples".into(),
        ));
    }
    for s in samples {
        if s.is_empty() {
            return Err(RsomicsError::InvalidInput(
                "anderson_ksamp encountered a sample without observations".into(),
            ));
        }
        if s.iter().any(|v| v.is_nan()) {
            return Err(RsomicsError::InvalidInput("input contains NaN".into()));
        }
    }

    // Z = sorted pooled; Zstar = sorted unique.
    let mut z: Vec<f64> = samples.iter().flatten().copied().collect();
    z.sort_by(|a, b| a.partial_cmp(b).expect("NaN guarded above"));
    let big_n = z.len();
    let zstar = unique_sorted(&z);
    if zstar.len() < 2 {
        return Err(RsomicsError::InvalidInput(
            "anderson_ksamp needs more than one distinct observation".into(),
        ));
    }

    let n: Vec<f64> = samples.iter().map(|s| s.len() as f64).collect();
    let sorted_samples: Vec<Vec<f64>> = samples
        .iter()
        .map(|s| {
            let mut v = s.clone();
            v.sort_by(|a, b| a.partial_cmp(b).expect("NaN guarded above"));
            v
        })
        .collect();

    let a2kn = if midrank {
        a2akn_midrank(&sorted_samples, &z, &zstar, &n, big_n)
    } else {
        a2kn_right(&sorted_samples, &z, &zstar, &n, big_n)
    };

    let a2 = normalize(&n, big_n, k, a2kn);
    let critical = ksamp_critical(k);
    let sig = vec![0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001];
    let (pvalue, p_clipped) = ksamp_pvalue(a2, &critical, &sig);

    Ok(KSampResult {
        statistic: a2,
        critical_values: critical,
        significance_level: sig,
        pvalue,
        p_clipped,
    })
}

/// `np.unique` on a sorted array: drop adjacent duplicates.
fn unique_sorted(z: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(z.len());
    for &v in z {
        if out.last().is_none_or(|&last| last != v) {
            out.push(v);
        }
    }
    out
}

/// `a.searchsorted(v, side)` for ascending `a`: count of elements that go before
/// `v`. 'left' counts elements `< v`, 'right' counts elements `<= v`.
fn searchsorted(a: &[f64], v: f64, right: bool) -> usize {
    let (mut lo, mut hi) = (0usize, a.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        let go_right = if right { a[mid] <= v } else { a[mid] < v };
        if go_right {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// A²akN, Scholz-Stephens eq. 7 (midrank, ties handled).
fn a2akn_midrank(samples: &[Vec<f64>], z: &[f64], zstar: &[f64], n: &[f64], big_n: usize) -> f64 {
    let nf = big_n as f64;
    let l = zstar.len();
    let no_ties = big_n == zstar.len();

    let mut bj = vec![0.0_f64; l];
    let mut lj = vec![0.0_f64; l];
    for (m, &zs) in zstar.iter().enumerate() {
        let left = searchsorted(z, zs, false) as f64;
        let ljm = if no_ties {
            1.0
        } else {
            searchsorted(z, zs, true) as f64 - left
        };
        lj[m] = ljm;
        bj[m] = left + ljm / 2.0;
    }

    let mut a2akn = 0.0;
    let mut inner = vec![0.0_f64; l];
    for (i, s) in samples.iter().enumerate() {
        for m in 0..l {
            let right = searchsorted(s, zstar[m], true) as f64;
            let fij = right - searchsorted(s, zstar[m], false) as f64;
            let mij = right - fij / 2.0;
            let num = lj[m] / nf * (nf * mij - bj[m] * n[i]).powi(2);
            let den = bj[m] * (nf - bj[m]) - nf * lj[m] / 4.0;
            inner[m] = num / den;
        }
        a2akn += crate::sum::pairwise_sum(&inner) / n[i];
    }
    a2akn * (nf - 1.0) / nf
}

/// A²kN, Scholz-Stephens eq. 6 (right variant).
fn a2kn_right(samples: &[Vec<f64>], z: &[f64], zstar: &[f64], n: &[f64], big_n: usize) -> f64 {
    let nf = big_n as f64;
    let l = zstar.len();
    // lj and Bj over zstar[:-1].
    let mut lj = vec![0.0_f64; l - 1];
    let mut bj = vec![0.0_f64; l - 1];
    let mut cum = 0.0;
    for m in 0..l - 1 {
        let ljm = searchsorted(z, zstar[m], true) as f64 - searchsorted(z, zstar[m], false) as f64;
        lj[m] = ljm;
        cum += ljm;
        bj[m] = cum;
    }

    let mut a2kn = 0.0;
    let mut inner = vec![0.0_f64; l - 1];
    for (i, s) in samples.iter().enumerate() {
        for m in 0..l - 1 {
            let mij = searchsorted(s, zstar[m], true) as f64;
            let num = lj[m] / nf * (nf * mij - bj[m] * n[i]).powi(2);
            let den = bj[m] * (nf - bj[m]);
            inner[m] = num / den;
        }
        a2kn += crate::sum::pairwise_sum(&inner) / n[i];
    }
    a2kn
}

/// Normalize A²kN to A² using the Scholz-Stephens H, h, g moments and σ².
fn normalize(n: &[f64], big_n: usize, k: usize, a2kn: f64) -> f64 {
    let nf = big_n as f64;
    let kf = k as f64;
    let cap_h: f64 = n.iter().map(|&ni| 1.0 / ni).sum();

    // hs_cs = cumsum(1/arange(N-1, 1, -1)); h = hs_cs[-1] + 1; g = sum(hs_cs/arange(2,N)).
    let mut hs_cs = Vec::with_capacity(big_n.saturating_sub(2));
    let mut running = 0.0;
    let mut denom = (big_n - 1) as f64;
    while denom > 1.0 {
        running += 1.0 / denom;
        hs_cs.push(running);
        denom -= 1.0;
    }
    let h = hs_cs.last().copied().unwrap_or(0.0) + 1.0;
    let g_terms: Vec<f64> = hs_cs
        .iter()
        .enumerate()
        .map(|(idx, &val)| val / (idx as f64 + 2.0))
        .collect();
    let g = crate::sum::pairwise_sum(&g_terms);

    let a = (4.0 * g - 6.0) * (kf - 1.0) + (10.0 - 6.0 * g) * cap_h;
    let b = (2.0 * g - 4.0) * kf * kf + 8.0 * h * kf + (2.0 * g - 14.0 * h - 4.0) * cap_h - 8.0 * h
        + 4.0 * g
        - 6.0;
    let c = (6.0 * h + 2.0 * g - 2.0) * kf * kf
        + (4.0 * h - 4.0 * g + 6.0) * kf
        + (2.0 * h - 6.0) * cap_h
        + 4.0 * h;
    let d = (2.0 * h + 6.0) * kf * kf - 4.0 * h * kf;
    let sigmasq =
        (a * nf.powi(3) + b * nf.powi(2) + c * nf + d) / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0));
    let m = kf - 1.0;
    (a2kn - m) / sigmasq.sqrt()
}

/// Tabulated k-sample critical values: `b0 + b1/√m + b2/m`, m = k−1.
fn ksamp_critical(k: usize) -> Vec<f64> {
    const B0: [f64; 7] = [0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085];
    const B1: [f64; 7] = [-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615];
    const B2: [f64; 7] = [-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154];
    let m = (k - 1) as f64;
    (0..7)
        .map(|i| B0[i] + B1[i] / m.sqrt() + B2[i] / m)
        .collect()
}

/// p-value via the probit-of-significance quadratic fit, capped/floored as scipy.
fn ksamp_pvalue(a2: f64, critical: &[f64], sig: &[f64]) -> (f64, bool) {
    let cmin = critical.iter().cloned().fold(f64::INFINITY, f64::min);
    let cmax = critical.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if a2 < cmin {
        return (sig.iter().cloned().fold(f64::NEG_INFINITY, f64::max), true);
    }
    if a2 > cmax {
        return (sig.iter().cloned().fold(f64::INFINITY, f64::min), true);
    }
    // pf = polyfit(critical, log(sig), 2); p = exp(polyval(pf, a2)).
    let logsig: Vec<f64> = sig.iter().map(|s| s.ln()).collect();
    let coeffs = polyfit2(critical, &logsig);
    let val = coeffs[0] * a2 * a2 + coeffs[1] * a2 + coeffs[2];
    (val.exp(), false)
}

/// Degree-2 least-squares polynomial fit, equivalent to `np.polyfit(x, y, 2)`.
/// Returns `[c2, c1, c0]` for `c2·x² + c1·x + c0`. Solves the 3×3 normal
/// equations the same way numpy's lstsq does for this well-conditioned system.
fn polyfit2(x: &[f64], y: &[f64]) -> [f64; 3] {
    let nrows = x.len();
    // Vandermonde columns are [x², x, 1]; build AᵀA (3×3) and Aᵀy.
    let mut ata = [[0.0_f64; 3]; 3];
    let mut aty = [0.0_f64; 3];
    for r in 0..nrows {
        let basis = [x[r] * x[r], x[r], 1.0];
        for (i, &bi) in basis.iter().enumerate() {
            for (j, &bj) in basis.iter().enumerate() {
                ata[i][j] += bi * bj;
            }
            aty[i] += bi * y[r];
        }
    }
    solve3(ata, aty)
}

/// Solve a 3×3 linear system by Gaussian elimination with partial pivoting.
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> [f64; 3] {
    for col in 0..3 {
        let mut piv = col;
        for r in col + 1..3 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        let pivot_row = a[col];
        for r in col + 1..3 {
            let f = a[r][col] / d;
            for (c, &pr) in pivot_row.iter().enumerate().skip(col) {
                a[r][c] -= f * pr;
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0_f64; 3];
    for r in (0..3).rev() {
        let mut s = b[r];
        for (c, &xc) in x.iter().enumerate().skip(r + 1) {
            s -= a[r][c] * xc;
        }
        x[r] = s / a[r][r];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn searchsorted_sides() {
        let a = [1.0, 2.0, 2.0, 3.0];
        assert_eq!(searchsorted(&a, 2.0, false), 1); // left: # < 2
        assert_eq!(searchsorted(&a, 2.0, true), 3); // right: # <= 2
        assert_eq!(searchsorted(&a, 0.0, false), 0);
        assert_eq!(searchsorted(&a, 5.0, true), 4);
    }

    #[test]
    fn unique_drops_dupes() {
        assert_eq!(
            unique_sorted(&[1.0, 1.0, 2.0, 3.0, 3.0]),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn rejects_single_sample() {
        assert!(anderson_ksamp(&[vec![1.0, 2.0]], true).is_err());
    }

    #[test]
    fn polyfit2_recovers_quadratic() {
        // y = 2x² - 3x + 1 sampled exactly -> coefficients recovered.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * v * v - 3.0 * v + 1.0).collect();
        let c = polyfit2(&x, &y);
        assert!((c[0] - 2.0).abs() < 1e-9);
        assert!((c[1] + 3.0).abs() < 1e-9);
        assert!((c[2] - 1.0).abs() < 1e-9);
    }
}
