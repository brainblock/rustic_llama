mod generator;
mod sampler;
mod tokenizer;
mod utils;

use rustic_llama::llama2::transformer::Transformer;
use sampler::Sampler;
use tokenizer::Tokenizer;
fn main() {
    let sampler = Sampler::new(0.0f32);
    let tokenizer: Tokenizer = Tokenizer::new("tokenizer.bin".into(), 32000).unwrap();
    let transformer = Transformer::new("stories15M.bin".into()).unwrap();
    generator::generate(&transformer, &tokenizer, &sampler, None, 300).unwrap();
}
