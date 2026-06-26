// Compute-only timing harness: preload the fixture, then time only the AD
// statistic over repeated calls — mirrors a scipy compute-only loop.
use std::time::Instant;

use rsomics_anderson_darling::{Dist, anderson, anderson_ksamp, parse_values};

fn load(path: &str) -> Vec<f64> {
    let f = std::fs::File::open(path).unwrap();
    parse_values(std::io::BufReader::new(f)).unwrap()
}

fn main() {
    let mode = std::env::args().nth(1).expect("mode: one|k");
    let reps = 50;
    if mode == "one" {
        let path = std::env::args().nth(2).expect("fixture path");
        let dist = Dist::parse(&std::env::args().nth(3).unwrap_or_else(|| "norm".into())).unwrap();
        let data = load(&path);
        let _ = anderson(&data, dist).unwrap();
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += anderson(&data, dist).unwrap().statistic;
        }
        let ms = t0.elapsed().as_secs_f64() / reps as f64 * 1000.0;
        eprintln!("compute-only one-sample mean: {ms:.3} ms over {reps} reps (acc={acc})");
    } else {
        let paths: Vec<String> = std::env::args().skip(2).collect();
        let samples: Vec<Vec<f64>> = paths.iter().map(|p| load(p)).collect();
        let _ = anderson_ksamp(&samples, true).unwrap();
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += anderson_ksamp(&samples, true).unwrap().statistic;
        }
        let ms = t0.elapsed().as_secs_f64() / reps as f64 * 1000.0;
        eprintln!("compute-only k-sample mean: {ms:.3} ms over {reps} reps (acc={acc})");
    }
}
