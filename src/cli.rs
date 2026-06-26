use std::fs::File;
use std::io::{self, BufReader};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use rsomics_common::{CommonFlags, RsomicsError, ToolMeta, run};
use serde::Serialize;

use rsomics_anderson_darling::{Dist, anderson, anderson_ksamp, parse_values};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

/// Anderson-Darling goodness-of-fit test (`scipy.stats.anderson` /
/// `anderson_ksamp`).
///
/// One-sample: a single column-of-values TSV plus `--dist`. The output is a
/// header line `# statistic<TAB>A2`, then a `significance<TAB>critical` table.
///
/// k-sample: two or more value TSVs with `--ksamp`. The output is one line,
/// `statistic<TAB>p`.
#[derive(Parser, Debug)]
#[command(name = "rsomics-anderson-darling", version, about, long_about = None)]
pub struct Cli {
    /// Input value TSV(s); one observation per line. One file = one-sample,
    /// two or more (with `--ksamp`) = k-sample. `-` reads stdin (one-sample).
    #[arg(value_name = "DATA", required = true)]
    pub data: Vec<PathBuf>,

    /// Reference distribution for the one-sample test.
    #[arg(long, value_name = "DIST", default_value = "norm")]
    pub dist: String,

    /// Run the k-sample test across the provided files.
    #[arg(long)]
    pub ksamp: bool,

    /// k-sample right variant (Scholz-Stephens eq. 6, no tie midranks). Without
    /// this flag the midrank variant (scipy default) is used. Ignored without
    /// `--ksamp`.
    #[arg(long)]
    pub right: bool,

    #[command(flatten)]
    pub common: CommonFlags,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Output {
    OneSample(rsomics_anderson_darling::AndersonResult),
    KSample(rsomics_anderson_darling::KSampResult),
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        run(&common, META, || {
            let output = if self.ksamp || self.data.len() > 1 {
                if self.data.len() < 2 {
                    return Err(RsomicsError::InvalidInput(
                        "--ksamp needs at least two sample files".into(),
                    ));
                }
                let mut samples = Vec::with_capacity(self.data.len());
                for p in &self.data {
                    let f = File::open(p).map_err(RsomicsError::Io)?;
                    samples.push(parse_values(BufReader::new(f))?);
                }
                let res = anderson_ksamp(&samples, !self.right)?;
                if !common.json {
                    println!("{}\t{}", res.statistic, res.pvalue);
                }
                Output::KSample(res)
            } else {
                let dist = Dist::parse(&self.dist)?;
                let values = match self.data.first() {
                    Some(p) if p.as_os_str() != "-" => {
                        let f = File::open(p).map_err(RsomicsError::Io)?;
                        parse_values(BufReader::new(f))?
                    }
                    _ => {
                        let stdin = io::stdin();
                        parse_values(stdin.lock())?
                    }
                };
                let res = anderson(&values, dist)?;
                if !common.json {
                    print_one_sample(&res);
                }
                Output::OneSample(res)
            };
            Ok(output)
        })
    }
}

fn print_one_sample(res: &rsomics_anderson_darling::AndersonResult) {
    println!("# dist\t{}", res.dist);
    println!("# statistic\t{}", res.statistic);
    println!("# pvalue\t{}", res.pvalue);
    println!("significance\tcritical");
    for (s, c) in res
        .significance_level
        .iter()
        .zip(res.critical_values.iter())
    {
        println!("{s}\t{c}");
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }
}
