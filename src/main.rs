mod sampler;
mod tokenizer;
mod generator;
mod utils;
mod transformer;

use sampler::Sampler;
use tokenizer::Tokenizer;
use transformer::Transformer;

fn main() {
    let sampler = Sampler::new(0.0f32);
    let tokenizer: Tokenizer = Tokenizer::new("tokenizer.bin".into(), 32000).unwrap();
    let mut transformer = Transformer::new("stories15M.bin".into()).unwrap();
    generator::generate(&mut transformer, &tokenizer, &sampler, None, 300).unwrap();

}


