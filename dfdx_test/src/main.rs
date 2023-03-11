#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
use dfdx::{prelude::*, nn::modules};

type Model<const VOCAB: usize, const EMBED: usize, const LAYERS: usize, const HEADS: usize, const SEQ: usize> = (
    Embedding<VOCAB, EMBED>,
    TransformerEncoder<EMBED, HEADS, {EMBED * 2}, LAYERS>,
    Linear<EMBED, VOCAB>,
);

type BuiltModel<const VOCAB: usize, const EMBED: usize, const LAYERS: usize, const HEADS: usize, const SEQ: usize, E, D> = (
    modules::Embedding<VOCAB, EMBED, E, D>,
    modules::TransformerEncoder<EMBED, HEADS, {EMBED * 2}, LAYERS, E, D>,
    modules::Linear<EMBED, VOCAB, E, D>,
);

// Model
const LAYERS: usize = 8;
const SEQ_LEN: usize = 25;
const EMBED_DIM: usize = 512;
const HEADS: usize = 8;

fn main() {
    let dev: Cuda = Default::default();
    let mut model = Model::<30527, EMBED_DIM, LAYERS, HEADS, SEQ_LEN>::build_on_device(&dev);
    test_epoch(&model, &dev)
}

fn test_epoch<const LAYERS: usize, const SEQ: usize, const VOCAB: usize, const EMBED: usize, const HEADS: usize, D: Device<f32>>(
    model: &BuiltModel<VOCAB, EMBED, LAYERS, HEADS, SEQ, f32, D>,
    dev: &D,
)
where 
    D: Device<f32>,
    BuiltModel<VOCAB, EMBED, LAYERS, HEADS, SEQ, f32, D>: Module<Tensor<(usize, Const<SEQ>), usize, D, NoneTape>, Output = Tensor<(usize, Const<SEQ>, Const<VOCAB>), f32, D, NoneTape>>,
    [(); EMBED * 2]:
{

}