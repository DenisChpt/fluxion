use thiserror::Error;

#[derive(Debug, Error)]
pub enum WgpuError {
	#[error("no GPU adapter available")]
	NoAdapter,

	#[error("device creation failed: {0}")]
	DeviceCreation(String),

	#[error("buffer map failed")]
	BufferMap,
}
