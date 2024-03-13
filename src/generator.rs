use std::io::Write;

use crate::{sampler::Sampler, tokenizer::Tokenizer, transformer::Transformer, utils::time_in_ms};
use anyhow::Result;

pub fn generate(
    transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &Sampler,
    prompt: Option<&str>,
    steps: usize,
) -> Result<()> {
    let prompt = prompt.unwrap_or("");

    let prompt_tokens = tokenizer.encode(prompt, true, false)?;

    // start the main loop
    let mut start = 0u128; // used to time our code, only initialized after first iteration
    let mut next: usize; // will store the next token in the sequence
    let mut token = prompt_tokens[0usize]; // kick off with the first token in the prompt
    let mut pos = 0usize; // position in the sequence
    println!("============");
    while pos < steps {
        // forward the transformer to get logits for the next token
        let logits = transformer.forward(token, pos);
        // advance the state machine
        if pos < prompt_tokens.len() - 1 {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sampler.sample(&logits);
        }
        pos += 1;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if next == 1 {
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        let piece = tokenizer.decode(token, next);
        safe_printf(&piece); // same as printf("%s", piece), but skips "unsafe" bytes
        token = next;

        // init the timer here because the first iteration can be slower
        if start == 0 {
            start = time_in_ms()?;
        }
    }
    print!("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if pos > 1 {
        let end = time_in_ms()?;
        eprintln!(
            "achieved tok/s: {}",
            (pos - 1) as f64 / (end - start) as f64 * 1000f64
        );
    }

    Ok(())
}

fn safe_printf(txt: &str) {
    print!("{}", txt);
    std::io::stdout().flush().unwrap();
}