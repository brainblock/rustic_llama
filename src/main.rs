use clap::Parser;

mod generator;
mod sampler;
mod tokenizer;
mod utils;

use rustic_llama::llama2::transformer::Transformer;
use sampler::Sampler;
use tokenizer::Tokenizer;

use humansize::{format_size, DECIMAL};

/// Rustic Lama
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model path.
    #[arg(short = 'p', long = "model_path", default_value_t = String::from("./stories15M.bin"))]
    model_path: String,

    /// Tokenizer path.
    #[arg(short = 't', long = "tokenizer_path", default_value_t = String::from("./tokenizer.bin"))]
    tokenizer_path: String,

    /// Temperature in search.
    #[arg(short = 'v', long = "temperature", default_value_t = 0.9)]
    temperature: f32,

    /// Number of steps in search.
    #[arg(short = 'n', long = "n_steps", default_value_t = 300)]
    n_steps: usize,

    /// Whether to use lazy model loading via mmap.
    #[arg(short = 'm', long = "is_mmap", default_value_t = false)]
    is_mmap: bool,

    /// Initial prompt
    #[arg(short = 's', long = "prompt", default_value_t = String::from(""))]
    prompt: String,
}

fn main() {
    let args = Args::parse();
    println!("using options: {:?}\n", args);


    let mut sampler = Sampler::new(args.temperature);
    let tokenizer: Tokenizer = Tokenizer::new(args.tokenizer_path.into(), 32000).unwrap();
    let transformer = Transformer::new(args.model_path.into()).unwrap();

    let transformer_size = datasize::data_size(&transformer);

    eprintln!("config: {:?}", transformer.config());
    eprintln!(
        "transformer_size heap: {}",
        format_size(transformer_size, DECIMAL)
    );
    generator::generate(&transformer, &tokenizer, &mut sampler, None, args.n_steps).unwrap();
}
