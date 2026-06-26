//! Normal CDF and log-CDF via Cephes/xsf ports.
//!
//! The Anderson-Darling statistic for `dist='norm'` needs `norm.logcdf` and
//! `norm.logsf`, which scipy routes to `scipy.special.log_ndtr` (Cephes `ndtr`
//! plus the xsf log-CDF asymptotics). Porting the same rational approximations
//! and region splits is what makes the A² statistic bit-identical to scipy.

// Coefficients are transcribed verbatim from Cephes at full source precision;
// digits past f64 precision round to the same bits.
#![allow(clippy::excessive_precision)]

const M_SQRT1_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
const MAXLOG: f64 = 7.097_827_128_933_840e2;
const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// Standard normal CDF Φ(a), Cephes `ndtr` (scipy `xsf` form, erf/erfc split at
/// `|a/√2| < 1`).
#[must_use]
pub fn ndtr(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = a * M_SQRT1_2;
    let z = x.abs();

    if z < 1.0 {
        0.5 + 0.5 * erf(x)
    } else {
        let y = 0.5 * erfc(z);
        if x > 0.0 { 1.0 - y } else { y }
    }
}

/// Standard normal log-CDF log Φ(a), matching `scipy.special.log_ndtr`.
///
/// Three regions, split at a=6 and a=-14: above 6 use `log1p(-ndtr(-a))` so the
/// upper tail keeps full precision; on `[-14, 6]` take `ln(ndtr(a))`; below -14
/// use the Mills-ratio asymptotic expansion in `1/a²`.
#[must_use]
pub fn log_ndtr(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    if a >= 6.0 {
        (-ndtr(-a)).ln_1p()
    } else if a >= -14.0 {
        ndtr(a).ln()
    } else {
        let t = 1.0 / (a * a);
        let series =
            1.0 - t * (1.0 - 3.0 * t * (1.0 - 5.0 * t * (1.0 - 7.0 * t * (1.0 - 9.0 * t))));
        -0.5 * a * a - LOG_SQRT_2PI - (-a).ln() + series.ln()
    }
}

/// Error function, Cephes `erf` — rational approximation for |x| < 1.
fn erf(x: f64) -> f64 {
    if x.abs() > 1.0 {
        return 1.0 - erfc(x);
    }
    let z = x * x;
    x * polevl(z, &T) / p1evl(z, &U)
}

/// Complementary error function, Cephes `erfc`.
fn erfc(a: f64) -> f64 {
    let x = a.abs();

    if x < 1.0 {
        return 1.0 - erf(a);
    }

    let z = -a * a;
    if z < -MAXLOG {
        return if a < 0.0 { 2.0 } else { 0.0 };
    }
    let z = z.exp();

    let (p, q) = if x < 8.0 {
        (polevl(x, &P), p1evl(x, &Q))
    } else {
        (polevl(x, &R), p1evl(x, &S))
    };
    let mut y = (z * p) / q;

    if a < 0.0 {
        y = 2.0 - y;
    }

    if y == 0.0 {
        return if a < 0.0 { 2.0 } else { 0.0 };
    }
    y
}

fn polevl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

fn p1evl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = x + coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

const T: [f64; 5] = [
    9.604_973_739_870_516_387_49e0,
    9.002_601_972_038_426_892_17e1,
    2.232_005_345_946_843_192_26e3,
    7.003_325_141_128_050_754_73e3,
    5.559_230_130_103_949_627_68e4,
];
const U: [f64; 5] = [
    3.356_171_416_475_030_996_47e1,
    5.213_579_497_801_526_797_95e2,
    4.594_323_829_709_801_279_87e3,
    2.262_900_006_138_909_342_46e4,
    4.926_739_426_086_359_210_86e4,
];

const P: [f64; 9] = [
    2.461_969_814_735_305_125_24e-10,
    5.641_895_648_310_688_219_77e-1,
    7.463_210_564_422_699_126_87e0,
    4.863_719_709_856_813_666_14e1,
    1.965_208_329_560_770_982_42e2,
    5.264_451_949_954_773_586_31e2,
    9.345_285_271_719_576_075_40e2,
    1.027_551_886_895_157_102_72e3,
    5.575_353_353_693_993_275_26e2,
];
const Q: [f64; 8] = [
    1.322_819_511_547_449_925_08e1,
    8.670_721_408_859_897_423_29e1,
    3.549_377_788_878_198_910_62e2,
    9.757_085_017_432_054_897_53e2,
    1.823_909_166_879_097_362_89e3,
    2.246_337_608_187_109_817_92e3,
    1.656_663_091_941_613_501_82e3,
    5.575_353_408_177_276_755_46e2,
];

const R: [f64; 6] = [
    5.641_895_835_477_550_739_84e-1,
    1.275_366_707_599_781_044_16e0,
    5.019_050_422_511_804_774_14e0,
    6.160_210_979_930_535_851_95e0,
    7.409_742_699_504_489_391_60e0,
    2.978_866_653_721_002_406_70e0,
];
const S: [f64; 6] = [
    2.260_528_632_201_172_765_90e0,
    9.396_035_249_380_014_346_73e0,
    1.204_895_398_080_966_566_05e1,
    1.708_144_507_475_658_972_22e1,
    9.608_968_090_632_858_781_98e0,
    3.369_076_451_000_815_160_50e0,
];

#[cfg(test)]
mod tests {
    use super::{log_ndtr, ndtr};

    fn rel(got: f64, want: f64) -> f64 {
        (got - want).abs() / want.abs().max(f64::MIN_POSITIVE)
    }

    // scipy.special.ndtr (scipy 1.17.1), both erf and erfc branches plus tails.
    const NDTR_GRID: &[(f64, f64)] = &[
        (-8.0, 6.22096057427174e-16),
        (-2.0, 0.022750131948179198),
        (-1.0, 0.15865525393145707),
        (-0.5, 0.3085375387259869),
        (0.0, 0.5),
        (0.5, 0.6914624612740131),
        (1.0, 0.8413447460685429),
        (2.0, 0.9772498680518208),
        (5.0, 0.9999997133484281),
    ];

    #[test]
    fn ndtr_matches_scipy() {
        for &(x, want) in NDTR_GRID {
            assert!(rel(ndtr(x), want) <= 1e-12, "ndtr({x})");
        }
    }

    // scipy.special.log_ndtr (scipy 1.17.1): the upper branch (>6), the central
    // ln(ndtr) branch, and the asymptotic tail (<-14).
    const LOG_NDTR_GRID: &[(f64, f64)] = &[
        (8.0, -6.220960574271742e-16),
        (6.0, -9.865876455243719e-10),
        (1.0, -0.1727537790234499),
        (0.5, -0.36894641528865635),
        (-2.0, -3.7831843336820317),
        (-6.0, -20.73676894997471),
        (-14.0, -101.5630344074499618),
        (-20.0, -203.91715537109727),
        (-40.0, -804.6084420137538),
        (-100.0, -5005.524208694205),
    ];

    #[test]
    fn log_ndtr_matches_scipy() {
        for &(x, want) in LOG_NDTR_GRID {
            let got = log_ndtr(x);
            // Near zero (deep upper tail, F≈1) the relative error of a ~1e-16
            // log value mirrors the 1-ULP noise of ndtr itself; bound it
            // absolutely there and relatively everywhere else.
            let ok = rel(got, want) <= 1e-12 || (got - want).abs() <= 1e-21;
            assert!(ok, "log_ndtr({x}) = {got:e} vs scipy {want:e}");
        }
    }
}
