use anyhow::Result;
use memmap2::{Mmap, MmapOptions};
use std::{fs::File, marker::PhantomData, mem};

use super::{config::Config, Llama2Weights};

struct MemoryMappedArray<T> {
    start: usize,
    length: usize,
    _t: PhantomData<T>,
}
impl<T> MemoryMappedArray<T> {
    fn new(start: usize, count: usize) -> Self {
        let tsize = mem::size_of::<T>();
        let length = (count) * tsize;
        Self {
            start,
            length,
            _t: PhantomData,
        }
    }

    #[inline]
    fn aligned<'b>(&self, bytes: &'b [u8]) -> &'b [T] {
        let size = mem::size_of::<T>();
        let len = self.length;
        let bytes = &bytes[self.start..self.end()];
        let ptr = bytes.as_ptr() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, len / size) }
    }

    fn end(&self) -> usize {
        self.start + self.length
    }
}
pub struct TransformerWeights {
    mmap: Mmap,
    // token embedding table
    token_embedding_table: MemoryMappedArray<f32>, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: MemoryMappedArray<f32>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: MemoryMappedArray<f32>, // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    wq: MemoryMappedArray<f32>, // (layer, dim, n_heads * head_size)
    wk: MemoryMappedArray<f32>, // (layer, dim, n_kv_heads * head_size)
    wv: MemoryMappedArray<f32>, // (layer, dim, n_kv_heads * head_size)
    wo: MemoryMappedArray<f32>, // (layer, n_heads * head_size, dim)
    // weights for ffn
    w1: MemoryMappedArray<f32>, // (layer, hidden_dim, dim)
    w2: MemoryMappedArray<f32>, // (layer, dim, hidden_dim)
    w3: MemoryMappedArray<f32>, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: MemoryMappedArray<f32>, // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    wcls: MemoryMappedArray<f32>,
}
impl TransformerWeights {
    pub fn from_bytes(config: &Config, f: File, start: usize) -> Result<Self> {
        let head_size = config.dim / config.n_heads;
        let mmap = unsafe { MmapOptions::new().map(&f)? };
        let token_embedding_table = MemoryMappedArray::new(start, config.vocab_size * config.dim);
        let rms_att_weight =
            MemoryMappedArray::new(token_embedding_table.end(), config.n_layers * config.dim);
        let wq = MemoryMappedArray::new(
            rms_att_weight.end(),
            config.n_layers * config.dim * (config.n_heads * head_size),
        );
        let wk = MemoryMappedArray::new(
            wq.end(),
            config.n_layers * config.dim * (config.n_kv_heads * head_size),
        );
        let wv = MemoryMappedArray::new(
            wk.end(),
            config.n_layers * config.dim * (config.n_kv_heads * head_size),
        );
        let wo = MemoryMappedArray::new(
            wv.end(),
            config.n_layers * (config.n_heads * head_size) * config.dim,
        );

        let rms_ffn_weight = MemoryMappedArray::new(wo.end(), config.n_layers * config.dim); // (layer, dim)
                                                                                             // weights for ffn
        let w1 = MemoryMappedArray::new(
            rms_ffn_weight.end(),
            config.n_layers * config.hidden_dim * config.dim,
        ); // (layer, hidden_dim, dim)
        let w2 = MemoryMappedArray::new(w1.end(), config.n_layers * config.dim * config.hidden_dim); // (layer, dim, hidden_dim)
        let w3 = MemoryMappedArray::new(w2.end(), config.n_layers * config.dim * config.hidden_dim); // (layer, hidden_dim, dim)
                                                                                                     // final rmsnorm
        let rms_final_weight = MemoryMappedArray::new(w3.end(), config.dim); // (dim,)

        // (optional) classifier weights for the logits, on the last layer
        // using shared weights
        let wcls = MemoryMappedArray::new(start, config.vocab_size * config.dim);
        Ok(Self {
            mmap,
            token_embedding_table,
            rms_att_weight,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_weight,
            w1,
            w2,
            w3,
            rms_final_weight,
            wcls,
        })
    }
}

impl Llama2Weights for TransformerWeights {
        #[inline]
    fn token_embedding_table(&self) -> &[f32] {
        self.token_embedding_table.aligned(self.mmap.as_ref())
    }
    #[inline]
    fn rms_att_weight(&self) -> &[f32] {
        self.rms_att_weight.aligned(self.mmap.as_ref())
    }
    #[inline]
    fn wq(&self) -> &[f32] {
        self.wq.aligned(self.mmap.as_ref())
    }
    #[inline]
    fn wk(&self) -> &[f32] {
        self.wk.aligned(self.mmap.as_ref())
    }
    #[inline]
    fn wv(&self) -> &[f32] {
        self.wv.aligned(self.mmap.as_ref())
    }
    #[inline]
    fn wo(&self) -> &[f32] {
        self.wo.aligned(self.mmap.as_ref())
    }
    #[inline]
    fn rms_ffn_weight(&self) -> &[f32] {
        self.rms_ffn_weight.aligned(self.mmap.as_ref())
    }
    #[inline]
    fn w1(&self) -> &[f32] {
        self.w1.aligned(self.mmap.as_ref())
    }
    #[inline]
    fn w2(&self) -> &[f32] {
        self.w2.aligned(self.mmap.as_ref())
    }
    #[inline]
    fn w3(&self) -> &[f32] {
        self.w3.aligned(self.mmap.as_ref())
    }
    #[inline]
    fn rms_final_weight(&self) -> &[f32] {
        self.rms_final_weight.aligned(self.mmap.as_ref())
    }
    #[inline]
    fn wcls(&self) -> &[f32] {
        self.wcls.aligned(self.mmap.as_ref())
    }
}
