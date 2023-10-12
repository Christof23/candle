use serde::Deserialize;
use std::path::PathBuf;

/// This is the configuration class to store the configuration of a [`NougatModel`]. It is used to
/// instantiate a Nougat model according to the specified arguments, defining the model architecture
///
/// Args:
///     input_size:
///         Input image size (canvas size) of Nougat.encoder, SwinTransformer in this codebase
///     align_long_axis:
///         Whether to rotate image if height is greater than width
///     window_size:
///         Window size of Nougat.encoder, SwinTransformer in this codebase
///     encoder_layer:
///         Depth of each Nougat.encoder Encoder layer, SwinTransformer in this codebase
///     decoder_layer:
///         Number of hidden layers in the Nougat.decoder, such as BART
///     max_position_embeddings
///         Trained max position embeddings in the Nougat decoder,
///         if not specified, it will have same value with max_length
///     max_length:
///         Max position embeddings(=maximum sequence length) you want to train
///     name_or_path:
///         Name of a pretrained model name either registered in huggingface.co. or saved in local

#[derive(Deserialize)]
struct NougatConfig {
    pub input_size: Vec<usize>,
    pub align_long_axis: bool,
    pub window_size: usize,
    pub encoder_layer: Vec<usize>,
    pub decoder_layer: usize,
    pub max_position_embeddings: Option<usize>,
    pub max_length: usize,
    pub name_or_path: PathBuf,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub num_heads: Vec<usize>,
    pub hidden_dimension: usize,
}

impl NougatConfig {
    pub fn default() -> Self {
        Self {
            input_size: vec![896, 672],
            align_long_axis: false,
            window_size: 7,
            encoder_layer: vec![2, 2, 14, 2],
            decoder_layer: 10,
            max_position_embeddings: None,
            max_length: 4096,
            name_or_path: PathBuf::new(),
            patch_size: 4,
            embed_dim: 128,
            num_heads: vec![4, 8, 16, 32],
            hidden_dimension: 1024,
        }
    }
}
