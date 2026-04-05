use thiserror::Error;

use crate::device::Device;

#[derive(Debug, Error)]
pub enum RuntimeError {
	#[error(transparent)]
	Core(#[from] fluxion_core::CoreError),

	#[error("device mismatch: {a} vs {b}")]
	DeviceMismatch { a: Device, b: Device },

	#[error("backend not available: {0}")]
	BackendUnavailable(String),
}

pub type Result<T> = std::result::Result<T, RuntimeError>;
