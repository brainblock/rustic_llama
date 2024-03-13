pub struct Sampler {
    temperature: f32,
}

impl Sampler {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }

    pub fn sample(&self, logits: &[f32]) -> usize {
        let next;
        if self.temperature == 0.0f32 {
            next = self.sample_argmax(logits)
        } else {
            unimplemented!()
        }
        next
    }

    fn sample_argmax(&self, probabilities: &[f32]) -> usize {
        // return the index that has the highest probability
        let mut max_i = 0;
        let mut max_p = probabilities.get(0).unwrap_or(&0f32).to_owned();

        for i in 0..probabilities.len() {
            if probabilities[i] > max_p {
                max_i = i;
                max_p = probabilities[i];
            }
        }
        max_i
    }
}

#[cfg(test)]
mod test {
    use crate::sampler::Sampler;

    #[test]
    fn test_sample_argmax() {

        let sampler = Sampler::new(0.0f32);
        let input = [0.3,0.8,0.2];
        let next = sampler.sample(&input);
        assert_eq!(next, 1);
    }
}
