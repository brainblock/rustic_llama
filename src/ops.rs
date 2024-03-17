// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer
use num_traits::{Float, Zero};
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

pub fn rmsnorm_self(o: &mut [f32], weight: &[f32]) {
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

pub fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32]) {
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

pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32]) {
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
pub fn softmax<T: Float + AddAssign + DivAssign>(x: &mut [T]) {
    // find max value (for numerical stability)
    let mut max_val = x[0];
    for i in 1..x.len() {
        if x[i] > max_val {
            max_val = x[i];
        }
    }
    // exp and sum
    let mut sum = T::zero();
    for i in 0..x.len() {
        x[i] = (x[i] - max_val).exp();
        sum += x[i];
    }
    // normalize
    for i in 0..x.len() {
        x[i] /= sum;
    }
}
pub fn accum<T: AddAssign + Copy>(a: &mut [T], b: &[T]) {
    debug_assert!(a.len() == b.len());
    for (ai, bi) in a.iter_mut().zip(b.iter()) {
        *ai += *bi;
    }
}

pub fn dotprod<T: AddAssign + Float>(a: &[T], b: &[T]) -> T {
    debug_assert!(a.len() == b.len());
    let mut v: T = Zero::zero();
    for (ai, bi) in a.iter().zip(b.iter()) {
        v += (*ai) * (*bi);
    }
    v
}

/// F.silu; silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
pub fn silu<T: MulAssign + Float>(x: &mut [T]) {
    for xi in x.iter_mut() {
        let one = T::one();
        *xi *= one / (one + (-(*xi)).exp());
    }
}

pub fn elementwise_mult<T: Float + MulAssign>(xout: &mut [T], x: &[T]) {
    for (xouti, xi) in xout.iter_mut().zip(x.iter()) {
        *xouti *= *xi;
    }
}

pub fn sample<T: Float + SubAssign>(a: &[T], rng: &mut rand::rngs::ThreadRng) -> usize
where
    Standard: Distribution<T>,
{
    debug_assert!(!a.is_empty());
    let mut v: T = rng.gen();
    for (i, ai) in a.iter().enumerate() {
        if v <= *ai {
            return i;
        }
        v -= *ai;
    }
    return a.len() - 1;
}

pub fn argmax<T: Float>(a: &[T]) -> usize {
    let mut max_idx: usize = 0;
    let mut max = T::neg_infinity();
    for (i, ai) in a.iter().enumerate() {
        if *ai > max {
            max = *ai;
            max_idx = i;
        }
    }
    max_idx
}

pub fn logits_to_prob<T: Float + DivAssign + AddAssign>(logits: &mut [T], temperature: T) {
    logits.iter_mut().for_each(|v| {
        *v /= temperature;
    });
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits);
}
