pub mod config;
pub mod transformer;
pub mod run_state;
pub(crate) mod weights;

pub trait Llama2Weights {
    fn token_embedding_table(&self) -> &[f32];
    fn rms_att_weight(&self) -> &[f32];
    fn wq(&self) -> &[f32];
    fn wk(&self) -> &[f32];
    fn wv(&self) -> &[f32];
    fn wo(&self) -> &[f32];
    fn rms_ffn_weight(&self) -> &[f32];
    fn w1(&self) -> &[f32];
    fn w2(&self) -> &[f32];
    fn w3(&self) -> &[f32];
    fn rms_final_weight(&self) -> &[f32];
    fn wcls(&self) -> &[f32];
}
