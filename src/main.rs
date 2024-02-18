mod sampler;
mod tokenizer;

use sampler::Sampler;
use tokenizer::Tokenizer;

fn main() {
    let sampler = Sampler::new(0.0f32);
    let tokenizer: Tokenizer = Tokenizer::new("tokenizer.bin".into(), 32000).unwrap();

    let encoded = tokenizer.encode("Once upon a time", true, false).unwrap();
    assert_eq!(encoded, vec![1,9038,2501,263,931])

}
