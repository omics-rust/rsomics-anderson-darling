//! Anderson-Darling goodness-of-fit tests — `scipy.stats.anderson` (one-sample)
//! and `scipy.stats.anderson_ksamp` (k-sample), value-exact to scipy 1.17.1.
//!
//! One-sample reports the A² statistic with scipy's tabulated critical values
//! and significance levels for `norm`/`expon`/`logistic`/`gumbel_l`/`gumbel_r`,
//! plus the interpolated p-value. k-sample reports the normalized A²kN statistic
//! and the interpolated/clipped p-value (midrank or right variant).

mod dist;
mod ksample;
mod ndtr;
mod onesample;
mod sum;

use std::io::BufRead;

use rsomics_common::{Result, RsomicsError};

pub use dist::Dist;
pub use ksample::{KSampResult, anderson_ksamp};
pub use onesample::{AndersonResult, anderson};

/// Parse a single-column numeric TSV (one value per line, blank lines skipped).
pub fn parse_values<R: BufRead>(reader: R) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(RsomicsError::Io)?;
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        // First whitespace/tab-delimited field, so two-column inputs still parse.
        let field = s.split_whitespace().next().unwrap_or(s);
        let v: f64 = field.parse().map_err(|_| {
            RsomicsError::InvalidInput(format!(
                "line {}: value '{field}' is not a number",
                lineno + 1
            ))
        })?;
        out.push(v);
    }
    if out.is_empty() {
        return Err(RsomicsError::InvalidInput(
            "no observations in input".into(),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_skips_blank_lines() {
        let input = "1.0\n\n2.5\n3\n";
        let v = parse_values(input.as_bytes()).unwrap();
        assert_eq!(v, vec![1.0, 2.5, 3.0]);
    }

    #[test]
    fn parse_rejects_non_numeric() {
        assert!(parse_values("1.0\nfoo\n".as_bytes()).is_err());
    }

    #[test]
    fn parse_rejects_empty() {
        assert!(parse_values("\n\n".as_bytes()).is_err());
    }
}
