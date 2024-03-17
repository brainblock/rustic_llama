use rand::thread_rng;
use rustic_llama::ops::{argmax, logits_to_prob, sample};

pub struct Sampler {
    temperature: f32,
    rng: rand::rngs::ThreadRng,
}

impl Sampler {
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature,
            rng: thread_rng(),
        }
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> usize {
        if self.temperature == 0.0f32 {
            argmax(logits)
        } else {
            logits_to_prob(logits, self.temperature);
            sample(logits, &mut self.rng)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::sampler::Sampler;

    #[test]
    fn test_sample_argmax() {
        let mut sampler = Sampler::new(0.0f32);
        let mut input = [0.3, 0.8, 0.2];
        let next = sampler.sample(&mut input);
        assert_eq!(next, 1);
    }
}
