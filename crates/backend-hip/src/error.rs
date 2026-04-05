/// HIP backend errors.
#[derive(Debug, thiserror::Error)]
pub enum HipError {
	#[error("HIP runtime error: {0}")]
	Runtime(String),

	#[error("no HIP device found")]
	NoDevice,

	#[error("kernel module not loaded: {0}")]
	ModuleLoad(String),

	#[error("kernel function not found: {0}")]
	KernelNotFound(String),
}

impl From<HipError> for fluxion_core::CoreError {
	fn from(e: HipError) -> Self {
		Self::BackendError(e.to_string())
	}
}
