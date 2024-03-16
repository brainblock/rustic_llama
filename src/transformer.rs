use anyhow::Result;
use memmap2::{Mmap, MmapOptions};
use std::{
    fs::File,
    io::{Cursor, Read},
    marker::PhantomData,
    mem,
    path::PathBuf,
};

use crate::utils::read_value;

#[derive(Debug)]
pub struct Config {
    dim: usize,        // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize,   // number of layers
    n_heads: usize,    // number of query heads
    n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    seq_len: usize,    // max sequence length
}

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
    fn from_bytes(config: &Config, f: File, start: usize) -> Result<Self> {
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
/// buffers for the "wave" of activations in the forward pass
pub struct RunState {
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
pub struct Transformer {
    config: Config,              // the hyperparameters of the architecture (the blueprint)
    weights: TransformerWeights, // the weights of the model
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

impl Transformer {
    pub fn new(checkpoint_path: PathBuf) -> Result<Self> {
        let f = File::open(checkpoint_path)?;
        let reader = unsafe { MmapOptions::new().map(&f)? };

        let mut cursor = Cursor::new(reader);

        let config = Config::from_read(&mut cursor)?;

        let start = cursor.position() as usize;
        let weights = TransformerWeights::from_bytes(&config, f, start)?;

        Ok(Self {
            config,
            weights,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
    
    pub fn weights(&self) -> &TransformerWeights {
        &self.weights
    }


}

impl RunState {
    pub fn new(config: &Config) -> Self {
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        Self {
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

    pub fn forward(&mut self, token: usize, pos: usize, p: &Config, w: &TransformerWeights) -> &[f32] {

        let dim = p.dim;
        let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        let kv_mul = p.n_heads / p.n_kv_heads;
        let hidden_dim = p.hidden_dim;
        let head_size = dim / p.n_heads;

        let token_emb = &w.token_embedding_table()[token * dim..(token + 1) * dim];
        self.x.copy_from_slice(token_emb);


        for l in 0..p.n_layers {
            rmsnorm(&mut self.xb, &self.x[..], &w.rms_att_weight()[l * dim..(l + 1) * dim]);

            // key and value point to the kv cache
            let loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience
            self.qkv(l, pos, p, w);
        
            self.rope_rotation(l, pos, p);

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
                    let mut score = 0.0f32;
                    for i in 0..head_size {
                        score += q[i] * k[i];
                    }
                    score /= (head_size as f32).sqrt();
                    // save the score to the attention buffer
                    att[t] = score;
                }
                softmax(att, pos + 1);

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
            for i in 0..dim {
                self.x[i] += self.xb2[i];
            }
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
            for i in 0..hidden_dim {
                let mut val = self.hb[i];
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                val *= 1.0f32 / (1.0f32 + (-val).exp());
                // elementwise multiply with w3(x)
                val *= self.hb2[i];
                self.hb[i] = val;
            }

            // final matmul to get the output of the ffn
            matmul(
                &mut self.xb[..],
                &self.hb[..],
                &w.w2()[l * dim * hidden_dim..(l + 1) * dim * hidden_dim],
            );

            // residual connection
            for i in 0..dim {
                self.x[i] += self.xb[i];
            }
        }
        // final rmsnorm
        rmsnorm_self(&mut self.x[..], &w.rms_final_weight()[..]);

        // classifier into logits
        matmul(&mut self.logits[..], &self.x[..], &w.wcls()[..]);

        &self.logits
    }

    fn qkv(&mut self, l: usize, pos: usize, p: &Config, w: &TransformerWeights) {
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

    fn rope_rotation(&mut self, l: usize, pos: usize, p: &Config) {
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
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

fn rmsnorm_self(o: &mut [f32], weight: &[f32]) {
    // calculate sum of squares
    let mut ss = 0.0f32;
    for v in o.iter() {
        ss += v * v;
    }
    ss /= o.len() as f32;
    ss += 1e-5f32;
    ss = 1.0f32 / ss.sqrt();
    // normalize and scale
    for j in 0..o.len() {
        o[j] = weight[j] * (ss * o[j]);
    }
}

fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32]) {
    // calculate sum of squares
    let mut ss = 0.0f32;
    for v in x {
        ss += v * v;
    }
    ss /= x.len() as f32;
    ss += 1e-5f32;
    ss = 1.0f32 / ss.sqrt();
    // normalize and scale
    for j in 0..x.len() {
        o[j] = weight[j] * (ss * x[j]);
    }
}

fn matmul(xout: &mut [f32], x: &[f32], w: &[f32]) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    let n = x.len();
    for i in 0..xout.len() {
        let mut val = 0.0f32;
        for j in 0..x.len() {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}
fn softmax(x: &mut [f32], size: usize) {
    // find max value (for numerical stability)
    let mut max_val = x[0];
    for i in 1..size {
        if x[i] > max_val {
            max_val = x[i];
        }
    }
    // exp and sum
    let mut sum = 0.0f32;
    for i in 0..x.len() {
        x[i] = (x[i] - max_val).exp();
        sum += x[i];
    }
    // normalize
    for i in 0..size {
        x[i] /= sum;
    }
}
// fn print_array(x: &[f32], desc: &str) {
//     let half = min(8usize, x.len())/2;
//     eprint!(" {}: [{}] : [", desc, x.len());
//     for i in 0..half {
//         eprint!("{}, ", x[i])
//     }
//     eprint!("... ");
//     for i in x.len()-half..x.len() {
//         eprint!("{}, ", x[i])
//     }
//     eprintln!("]");
// }
