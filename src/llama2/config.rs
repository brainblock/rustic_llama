use std::io::Read;

use anyhow::Result;

use crate::utils::read_value;

#[derive(Debug)]
pub struct Config {
    pub dim: usize,        // transformer dimension
    pub hidden_dim: usize, // for ffn layers
    pub n_layers: usize,   // number of layers
    pub n_heads: usize,    // number of query heads
    pub n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    pub seq_len: usize,    // max sequence length
}

impl Config {
    pub fn from_read(mut reader: &mut impl Read) -> Result<Self> {
        Ok(Self {
            dim: read_value::<i32>(&mut reader)? as usize,
            hidden_dim: read_value::<i32>(&mut reader)? as usize,
            n_layers: read_value::<i32>(&mut reader)? as usize,
            n_heads: read_value::<i32>(&mut reader)? as usize,
            n_kv_heads: read_value::<i32>(&mut reader)? as usize,
            vocab_size: read_value::<i32>(&mut reader)? as usize,
            seq_len: read_value::<i32>(&mut reader)? as usize,
        })
    }
}