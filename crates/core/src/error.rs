use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
	#[error(
		"dimension mismatch: expected {expected}, got {got}"
	)]
	DimensionMismatch { expected: usize, got: usize },

	#[error("invalid grid: {0}")]
	InvalidGrid(String),

	#[error("invalid stencil: {0}")]
	InvalidStencil(String),

	#[error("backend error: {0}")]
	BackendError(String),
}

pub type Result<T> = std::result::Result<T, CoreError>;

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn dimension_mismatch_display() {
		let e = CoreError::DimensionMismatch {
			expected: 100,
			got: 50,
		};
		let msg = e.to_string();
		assert!(msg.contains("100"));
		assert!(msg.contains("50"));
	}

	#[test]
	fn invalid_grid_display() {
		let e =
			CoreError::InvalidGrid("too small".into());
		assert!(e.to_string().contains("too small"));
	}

	#[test]
	fn invalid_stencil_display() {
		let e = CoreError::InvalidStencil(
			"empty entries".into(),
		);
		assert!(e.to_string().contains("empty entries"));
	}
}
