use thiserror::Error;

#[derive(Error, Debug)]
pub enum VocalisError {
    #[error("Model error: {0}")]
    ModelError(String),
    #[error("Audio processing error: {0}")]
    AudioError(String),
    #[error("Input data error: {0}")]
    InputError(String),
    #[error("Prediction error: {0}")]
    PredictionError(String),
    #[error("Other error: {0}")]
    Other(String),
}

// Helper to convert ort::Error to VocalisError::ModelError
impl From<ort::Error> for VocalisError {
    fn from(err: ort::Error) -> Self {
        VocalisError::ModelError(err.to_string())
    }
}

// Helper to convert ndarray::ShapeError to VocalisError::Other
impl From<ndarray::ShapeError> for VocalisError {
    fn from(err: ndarray::ShapeError) -> Self {
        VocalisError::Other(err.to_string())
    }
}

// Helper to convert anyhow::Error to VocalisError::Other
impl From<anyhow::Error> for VocalisError {
    fn from(err: anyhow::Error) -> Self {
        VocalisError::Other(err.to_string())
    }
}
