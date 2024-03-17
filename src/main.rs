mod generator;
mod sampler;
mod tokenizer;
mod utils;

use rustic_llama::llama2::transformer::Transformer;
use sampler::Sampler;
use tokenizer::Tokenizer;

use humansize::{format_size, DECIMAL};

fn main() {
    let mut sampler = Sampler::new(0.0f32);
    let tokenizer: Tokenizer = Tokenizer::new("tokenizer.bin".into(), 32000).unwrap();
    let transformer = Transformer::new("stories15M.bin".into()).unwrap();

    let transformer_size = datasize::data_size(&transformer);
    eprintln!("config: {:?}", transformer.config());
    eprintln!(
        "transformer_size heap: {}",
        format_size(transformer_size, DECIMAL)
    );
    generator::generate(&transformer, &tokenizer, &mut sampler, None, 300).unwrap();
}
