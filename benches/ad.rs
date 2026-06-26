use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use rsomics_anderson_darling::{Dist, anderson, anderson_ksamp};

fn synth(n: usize, seed: u64) -> Vec<f64> {
    // Cheap deterministic LCG mapped through a Box-Muller-ish transform.
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let u1 = next().max(1e-12);
            let u2 = next();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        })
        .collect()
}

fn bench_anderson(c: &mut Criterion) {
    let data = synth(100_000, 42);
    c.bench_function("anderson_norm_100k", |b| {
        b.iter(|| anderson(black_box(&data), Dist::Norm).unwrap());
    });

    let s1 = synth(40_000, 1);
    let s2 = synth(35_000, 2);
    let s3 = synth(45_000, 3);
    let samples = vec![s1, s2, s3];
    c.bench_function("anderson_ksamp_120k", |b| {
        b.iter(|| anderson_ksamp(black_box(&samples), true).unwrap());
    });
}

criterion_group!(benches, bench_anderson);
criterion_main!(benches);
