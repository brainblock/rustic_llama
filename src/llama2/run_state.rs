use datasize::{data_size, DataSize};

use crate::ops::{accum, dotprod, elementwise_mult, matmul, rmsnorm, silu, softmax};

use super::{config::Config, Llama2Weights};

/// buffers for the "wave" of activations in the forward pass
pub struct RunState<'a, W: Llama2Weights> {
    config: &'a Config,
    weights: &'a W,
    pos: usize,
    // current wave of activations
    x: Vec<f32>,   // activation at current time stamp (dim,)
    xb: Vec<f32>,  // same, but inside a residual branch (dim,)
    xb2: Vec<f32>, // an additional buffer just for convenience (dim,)
    hb: Vec<f32>,  // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>, // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f32>,   // query (dim,)
    //k: Vec<f32>,      // key (dim,) only using kv_cache
    //v: Vec<f32>,      // value (dim,) only using kv_cache
    att: Vec<f32>,    // buffer for scores/attention values (n_heads, seq_len)
    logits: Vec<f32>, // output logits
    //kv cache
    key_cache: Vec<f32>,   // (layer, seq_len, dim)
    value_cache: Vec<f32>, // (layer, seq_len, dim)
}

impl<'a, W: Llama2Weights> DataSize for RunState<'a, W> {
    // `MyType` contains a `Vec`, so `IS_DYNAMIC` is set to true.
    const IS_DYNAMIC: bool = true;

    // The only always present heap item is the `counter` value, which is 8 bytes.
    const STATIC_HEAP_SIZE: usize = 8;

    #[inline]
    fn estimate_heap_size(&self) -> usize {
        // We can be lazy here and delegate to all the existing implementations:
        data_size(&self.pos)
            + data_size(&self.x)
            + data_size(&self.xb)
            + data_size(&self.xb2)
            + data_size(&self.hb)
            + data_size(&self.hb2)
            + data_size(&self.q)
            + data_size(&self.att)
            + data_size(&self.logits)
            + data_size(&self.key_cache)
            + data_size(&self.value_cache)
    }
}
impl<'a, W: Llama2Weights> RunState<'a, W> {
    pub fn new(config: &'a Config, weights: &'a W) -> Self {
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        Self {
            config,
            weights,
            pos: 0usize,
            x: vec![0f32; config.dim],
            xb: vec![0f32; config.dim],
            xb2: vec![0f32; config.dim],
            hb: vec![0f32; config.hidden_dim],
            hb2: vec![0f32; config.hidden_dim],
            q: vec![0f32; config.dim],
            // k: vec![0f32; config.dim],
            // v: vec![0f32; config.dim],
            att: vec![0f32; config.n_heads * config.seq_len],
            logits: vec![0f32; config.vocab_size],
            key_cache: vec![0f32; config.n_layers * config.seq_len * kv_dim],
            value_cache: vec![0f32; config.n_layers * config.seq_len * kv_dim],
        }
    }

    pub fn forward(&mut self, token: usize) -> &mut [f32] {
        let p = self.config;
        let dim = p.dim;
        let w = self.weights;

        let token_emb = &w.token_embedding_table()[token * dim..(token + 1) * dim];
        self.x.copy_from_slice(token_emb);

        for l in 0..p.n_layers {
            rmsnorm(
                &mut self.xb,
                &self.x[..],
                &w.rms_att_weight()[l * dim..(l + 1) * dim],
            );

            // key and value point to the kv cache
            self.qkv(l);

            self.rope_rotation(l);

            self.multihead_attention(l);

            self.fnn(l);
        }
        // final rmsnorm
        // rmsnorm_self(&mut self.x[..], &w.rms_final_weight()[..]);
        self.xb2.copy_from_slice(&self.x); // Temp copy x to xb2, and used for rmsnorm below.
        rmsnorm(&mut self.x[..], &self.xb2[..], w.rms_final_weight());

        // classifier into logits
        matmul(&mut self.logits[..], &self.x[..], &w.wcls()[..]);
        self.pos += 1;

        &mut self.logits
    }

    fn qkv(&mut self, l: usize) {
        let pos = self.pos;
        let p = self.config;
        let w = self.weights;
        let dim = p.dim;
        let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;

        // key and value point to the kv cache
        let loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience
        let mut q = &mut self.q[..];
        let mut k = &mut self.key_cache[loff + pos * kv_dim..loff + dim + pos * kv_dim];
        let mut v = &mut self.value_cache[loff + pos * kv_dim..loff + dim + pos * kv_dim];
        let wq = &w.wq()[l * dim * dim..(l + 1) * dim * dim];
        let wk = &w.wk()[l * dim * kv_dim..(l + 1) * dim * kv_dim];
        let wv = &w.wv()[l * dim * kv_dim..(l + 1) * dim * kv_dim];
        // let k = &s.key_cache[ loff + pos * kv_dim];

        // s.v = s.value_cache + loff + pos * kv_dim;
        matmul(&mut q, &self.xb, &wq);
        matmul(&mut k, &self.xb, &wk);
        matmul(&mut v, &self.xb, &wv);
    }

    fn rope_rotation(&mut self, l: usize) {
        let pos = self.pos;
        let p = self.config;
        let dim = p.dim;
        let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        let head_size = dim / p.n_heads;
        // key and value point to the kv cache
        let loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience
        let mut q = &mut self.q[..];
        let mut k = &mut self.key_cache[loff + pos * kv_dim..loff + dim + pos * kv_dim];

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for i in (0..dim).step_by(2) {
            let head_dim = i % head_size;
            let freq = 1.0f32 / 10000.0f32.powf(head_dim as f32 / head_size as f32);
            let val = pos as f32 * freq;
            let fcr = val.cos();
            let fci = val.sin();
            let rotn = if i < kv_dim { 2 } else { 1 }; // how many vectors? 2 = q & k, 1 = q only
            for v in 0..rotn {
                let vec = if v == 0 { &mut q } else { &mut k }; // the vector to rotate (query or key)
                let v0 = vec[i];
                let v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
    }
    fn multihead_attention(&mut self, l: usize) {
        let pos = self.pos;
        let p = self.config;
        let w = self.weights;
        let dim = p.dim;
        let head_size = dim / p.n_heads;
        let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        let kv_mul = p.n_heads / p.n_kv_heads;
        let loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience
                                           // multihead attention. iterate over all heads
        for h in 0..p.n_heads {
            // get the query vector for this head
            let q = &mut self.q[h * head_size..(h + 1) * head_size];
            // attention scores for this head
            let att = &mut self.att[h * p.seq_len..h * p.seq_len + pos + 1];
            // iterate over all timesteps, including the current one
            for t in 0..(pos + 1) {
                // get the key vector for this head and at this timestep
                let start_slice = loff + t * kv_dim + (h / kv_mul) * head_size;
                let end_slice = start_slice + head_size;
                let k = &mut self.key_cache[start_slice..end_slice];

                // calculate the attention score as the dot product of q and k
                let mut score = dotprod(q, k);
                score /= (head_size as f32).sqrt();
                // save the score to the attention buffer
                att[t] = score;
            }
            softmax(att);

            // weighted sum of the values, store back into xb
            let xb = &mut self.xb[h * head_size..(h + 1) * head_size];
            xb.fill(0f32);
            for t in 0..pos + 1 {
                // get the value vector for this head and at this timestep
                let start_slice = loff + t * kv_dim + (h / kv_mul) * head_size;
                let end_slice = start_slice + head_size;
                let v = &mut self.value_cache[start_slice..end_slice];

                // get the attention weight for this timestep
                let a = att[t];
                // accumulate the weighted value into xb
                for i in 0..head_size {
                    xb[i] += a * v[i];
                }
            }
        }
        // final matmul to get the output of the attention
        let wo = &w.wo()[l * dim * dim..(l + 1) * dim * dim];
        matmul(&mut self.xb2[..], &self.xb[..], &wo);

        // residual connection back into x
        accum(&mut self.x, &self.xb2);
    }
    // FFN calculates: self.w2(F.silu(self.w1(x)) * self.w3(x)).
    fn fnn(&mut self, l: usize) {
        let p = self.config;
        let w = self.weights;
        let dim = p.dim;
        let hidden_dim = p.hidden_dim;
        // ffn rmsnorm
        rmsnorm(
            &mut self.xb[..],
            &self.x[..],
            &w.rms_ffn_weight()[l * dim..(l + 1) * dim],
        );

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(
            &mut self.hb[..],
            &self.xb[..],
            &w.w1()[l * dim * hidden_dim..(l + 1) * dim * hidden_dim],
        );
        matmul(
            &mut self.hb2[..],
            &self.xb[..],
            &w.w3()[l * dim * hidden_dim..(l + 1) * dim * hidden_dim],
        );

        // SwiGLU non-linearity
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        silu(&mut self.hb);
        // elementwise multiply with w3(x)
        elementwise_mult(&mut self.hb, &self.hb2);

        // final matmul to get the output of the ffn
        matmul(
            &mut self.xb[..],
            &self.hb[..],
            &w.w2()[l * dim * hidden_dim..(l + 1) * dim * hidden_dim],
        );

        // residual connection
        accum(&mut self.x, &self.xb);
    }
}
