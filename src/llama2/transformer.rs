use anyhow::Result;
use datasize::DataSize;
use memmap2::MmapOptions;
use std::{fs::File, io::Cursor, path::PathBuf};

use super::{config::Config, run_state::RunState, weights::TransformerWeights};

#[derive(DataSize)]
pub struct Transformer {
    config: Config,              // the hyperparameters of the architecture (the blueprint)
    weights: TransformerWeights, // the weights of the model
}

impl Transformer {
    pub fn new(checkpoint_path: PathBuf) -> Result<Self> {
        let f = File::open(checkpoint_path)?;
        let reader = unsafe { MmapOptions::new().map(&f)? };

        let mut cursor = Cursor::new(reader);

        let config = Config::from_read(&mut cursor)?;

        let start = cursor.position() as usize;
        let weights = TransformerWeights::from_bytes(&config, f, start)?;

        Ok(Self { config, weights })
    }

    pub fn create_run_state(&self) -> RunState<TransformerWeights> {
        RunState::new(&self.config, &self.weights)
    }
    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn weights(&self) -> &TransformerWeights {
        &self.weights
    }
}

// fn print_array(x: &[f32], desc: &str) {
//     let half = min(8usize, x.len())/2;
//     eprint!(" {}: [{}] : [", desc, x.len());
//     for i in 0..half {
//         eprint!("{}, ", x[i])
//     }
//     eprint!("... ");
//     for i in x.len()-half..x.len() {
//         eprint!("{}, ", x[i])
//     }
//     eprintln!("]");
// }
