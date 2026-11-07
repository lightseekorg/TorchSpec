//! Qwen3-VL image preprocessing using the pinned upstream SMG processor.

use std::{fs, io::Cursor, path::Path, sync::Arc};

use image::ImageReader;
use llm_multimodal::{
    vision::PreProcessorConfig, EncoderDtype, ImageDetail, ImageFrame, ImageSize, ImageSource,
    Modality, ModelMetadata, ModelRegistry, ModelSpecificValue, TransparencyPolicy,
    VisionPreProcessor,
};
use llm_tokenizer::{create_tokenizer_with_chat_template, TokenizerTrait};
use pyo3::{prelude::*, types::PyBytes};
use serde_json::Value;

use crate::value_error;

struct RegistryTokenizer(Arc<dyn TokenizerTrait>);

impl llm_multimodal::Tokenizer for RegistryTokenizer {
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }
    fn encode_text(&self, text: &str) -> Option<Vec<u32>> {
        self.0
            .encode(text, false)
            .ok()
            .map(|e| e.token_ids().to_vec())
    }
}

#[pyclass(frozen)]
pub struct ImageProcessor {
    processor: Box<dyn VisionPreProcessor>,
    tokenizer: RegistryTokenizer,
    config: Value,
}

type ImageOutput = (Py<PyBytes>, Vec<usize>, Vec<i64>, Vec<Vec<i32>>);

#[pymethods]
impl ImageProcessor {
    #[new]
    fn new(py: Python<'_>, path: &str) -> PyResult<Self> {
        py.detach(|| {
            let root = Path::new(path);
            let config: Value =
                serde_json::from_slice(&fs::read(root.join("config.json")).map_err(value_error)?)
                    .map_err(value_error)?;
            if config["model_type"] != "qwen3_vl" {
                return Err(value_error(
                    "SMG image training currently requires model_type=qwen3_vl",
                ));
            }
            let preprocessor = PreProcessorConfig::from_json(
                &fs::read_to_string(root.join("preprocessor_config.json")).map_err(value_error)?,
            )
            .map_err(value_error)?;
            let tokenizer = RegistryTokenizer(
                create_tokenizer_with_chat_template(path, None).map_err(value_error)?,
            );
            let registry = ModelRegistry::new();
            let metadata = ModelMetadata {
                model_id: "",
                tokenizer: &tokenizer,
                config: &config,
            };
            let spec = registry
                .lookup(&metadata)
                .ok_or_else(|| value_error("No SMG vision spec"))?;
            let processor = spec
                .vision_processor(&config, &preprocessor, Modality::Image)
                .map_err(value_error)?;
            Ok(Self {
                processor,
                tokenizer,
                config,
            })
        })
    }

    fn process(&self, py: Python<'_>, images: Vec<Vec<u8>>) -> PyResult<ImageOutput> {
        let (pixels, shape, grid, replacements) = py.detach(|| {
            if images.is_empty() {
                return Err(value_error(
                    "Image preprocessing requires at least one image",
                ));
            }
            let decoded = images
                .into_iter()
                .map(|data| {
                    let (width, height) = ImageReader::new(Cursor::new(&data))
                        .with_guessed_format()
                        .map_err(value_error)?
                        .into_dimensions()
                        .map_err(value_error)?;
                    let frame = ImageFrame::encoded(
                        bytes::Bytes::from(data),
                        ImageDetail::Auto,
                        ImageSource::InlineBytes,
                        String::new(),
                        ImageSize::new(width, height),
                        None,
                    )
                    .with_transparency(TransparencyPolicy::CompositeOverWhite);
                    frame.image().cloned().map_err(value_error)
                })
                .collect::<PyResult<Vec<_>>>()?;
            let processed = self.processor.preprocess(&decoded).map_err(value_error)?;
            let grid = match processed.model_specific.get("image_grid_thw") {
                Some(ModelSpecificValue::IntTensor { data, shape })
                    if shape == &[decoded.len(), 3] =>
                {
                    data.clone()
                }
                _ => return Err(value_error("SMG returned an invalid image_grid_thw")),
            };
            let registry = ModelRegistry::new();
            let metadata = ModelMetadata {
                model_id: "",
                tokenizer: &self.tokenizer,
                config: &self.config,
            };
            let spec = registry
                .lookup(&metadata)
                .ok_or_else(|| value_error("No SMG vision spec"))?;
            let replacements = spec
                .prompt_replacements(&metadata, &processed)
                .map_err(value_error)?;
            let replacements = replacements
                .into_iter()
                .map(|r| r.tokens)
                .collect::<Vec<_>>();
            Ok((
                processed.encoder_input.write_as(EncoderDtype::Float32),
                processed.encoder_input_shape(),
                grid,
                replacements,
            ))
        })?;
        Ok((
            PyBytes::new(py, &pixels).unbind(),
            shape,
            grid,
            replacements,
        ))
    }
}
