use fluxion_core::{BackendBuffer, DType};

/// CPU buffer — contiguous heap allocation, no resizing.
#[derive(Debug)]
pub struct CpuBuffer {
	data: Box<[f64]>,
	dtype: DType,
}

impl CpuBuffer {
	#[inline]
	pub(crate) fn zeros(len: usize, dtype: DType) -> Self {
		Self {
			data: vec![0.0; len].into_boxed_slice(),
			dtype,
		}
	}

	#[inline]
	pub(crate) fn from_f64(data: &[f64]) -> Self {
		Self {
			data: data.into(),
			dtype: DType::F64,
		}
	}

	#[inline]
	pub(crate) fn from_f32(data: &[f32]) -> Self {
		Self {
			data: data.iter().map(|&v| f64::from(v)).collect(),
			dtype: DType::F32,
		}
	}

	#[inline]
	#[must_use]
	pub fn as_slice(&self) -> &[f64] {
		&self.data
	}

	#[inline]
	pub fn as_mut_slice(&mut self) -> &mut [f64] {
		&mut self.data
	}
}

impl BackendBuffer for CpuBuffer {
	#[inline]
	fn len(&self) -> usize {
		self.data.len()
	}

	#[inline]
	fn dtype(&self) -> DType {
		self.dtype
	}

	fn copy_to_host_f64(&self, dst: &mut [f64]) {
		assert_eq!(
			dst.len(),
			self.data.len(),
			"destination length mismatch"
		);
		dst.copy_from_slice(&self.data);
	}

	fn copy_to_host_f32(&self, dst: &mut [f32]) {
		assert_eq!(
			dst.len(),
			self.data.len(),
			"destination length mismatch"
		);
		for (d, &s) in dst.iter_mut().zip(self.data.iter()) {
			*d = s as f32;
		}
	}
}
