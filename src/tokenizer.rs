use memmap2::MmapOptions;

use anyhow::{Ok, Result};
use regex::Regex;
use std::io::{Cursor, Read};
use std::{fs::File, path::PathBuf};

use crate::utils::read_value;

struct TokenIndex {
    str: String,
    id: usize,
}

pub struct Tokenizer {
    vocab: Vec<String>,
    sorted_vocab: Vec<TokenIndex>,
    vocab_scores: Vec<f32>,
    byte_pieces: Vec<String>,
    // max_token_length: usize,
}

impl Tokenizer {
    pub fn new(tokenizer_path: PathBuf, vocab_size: usize) -> Result<Self> {
        let f = File::open(tokenizer_path)?;
        let reader = unsafe { MmapOptions::new().map(&f)? };

        let mut buf = Cursor::new(reader);

        let mut vocab: Vec<String> = Vec::with_capacity(vocab_size);
        let mut vocab_scores: Vec<f32> = Vec::with_capacity(vocab_size);
        let mut byte_pieces: Vec<String> = Vec::with_capacity(256);

        let mut sorted_vocab: Vec<TokenIndex> = Vec::with_capacity(vocab_size);

        let _max_token_length = read_value::<i32>(&mut buf)? as usize;

        for i in 0..255u8 {
            byte_pieces.push(format!("{}", i as char))
        }

        for i in 0..vocab_size {
            let score = read_value::<f32>(&mut buf)?;
            vocab_scores.push(score);

            let len = read_value::<i32>(&mut buf)? as usize;

            let mut bytes = vec![0; len];
            buf.read(bytes.as_mut_slice())?;

            let vocab_entry = String::from_utf8(bytes)?;
            vocab.push(vocab_entry.clone());

            sorted_vocab.push(TokenIndex {
                id: i,
                str: vocab_entry,
            });
        }
        sorted_vocab.sort_by(|a, b| a.str.cmp(&b.str));

        Ok(Tokenizer {
            vocab,
            vocab_scores,
            byte_pieces,
            sorted_vocab,
            // max_token_length,
        })
    }

    pub fn decode(&self, prev_token: usize, token: usize) -> &str {
        let mut piece: &str = &self.vocab[token];
        if prev_token == 1 && piece.starts_with(" ") {
            piece = piece.trim_start();
        }

        if piece.contains("<0x") {
            let re = Regex::new(r"<0x(.+)>$").unwrap();
            let captured = re.captures(&piece).take().unwrap();
            let byte_val = captured.get(1).unwrap();
            let byte_val: u8 = u8::from_str_radix(byte_val.as_str(), 16).unwrap();
            piece = &self.byte_pieces[byte_val as usize];
        }

        piece
    }

    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Result<Vec<usize>> {
        let mut tokens = Vec::with_capacity(text.len());

        // add optional BOS (=1) token, if desired
        if bos {
            tokens.push(1);
        }

        if text.chars().nth(0) != Some('\0') {
            let dummy_prefix = self.str_lookup(" ").expect("dummy prefix");

            tokens.push(dummy_prefix);
        }

        for character in text.chars() {
            if let Result::Ok(id) = self.str_lookup(&format!("{}", character)) {
                tokens.push(id);
            } else {
                // byte_fallback encoding
                unimplemented!("byte_fallback encoding")
            }
        }
        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        struct BestScore {
            id: usize,
            idx: usize,
        }
        loop {
            let mut best_score = f32::NEG_INFINITY;

            let mut best: Option<BestScore> = None;

            // check if we can merge the pair
            for (idx, pair) in tokens.windows(2).enumerate() {
                let str_buff = format!("{}{}", self.vocab[pair[0]], self.vocab[pair[1]]);

                if let Result::Ok(id) = self.str_lookup(&str_buff) {
                    if self.vocab_scores[id] > best_score {
                        best_score = self.vocab_scores[id];
                        best = Some(BestScore { id, idx });
                    }
                }
            }
            if let Some(best) = best {
                // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
                tokens[best.idx] = best.id;
                tokens.remove(best.idx + 1);
            } else {
                break; // we couldn't find any more pairs to merge, so we're done
            }
        }
        // add optional EOS (=2) token, if desired
        if eos {
            tokens.push(2);
        }

        Ok(tokens)
    }

    fn str_lookup(&self, str: &str) -> Result<usize, usize> {
        self.sorted_vocab
            .binary_search_by(|probe| probe.str.as_str().cmp(str))
            .map(|result| self.sorted_vocab[result].id)
    }
}

#[cfg(test)]
mod test {
    use crate::tokenizer::Tokenizer;

    #[test]
    fn test_simple_encode() {
        let tokenizer = Tokenizer::new("tokenizer.bin".into(), 32000).unwrap();

        let encoded = tokenizer.encode("Once upon a time", true, false).unwrap();

        assert_eq!(encoded, vec![1, 9038, 2501, 263, 931])
    }
}
