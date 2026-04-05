/// Identifies which compute device a buffer lives on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
	Cpu,
	#[cfg(feature = "wgpu")]
	Wgpu { adapter: usize },
	#[cfg(feature = "cuda")]
	Cuda { ordinal: usize },
	#[cfg(feature = "hip")]
	Hip { ordinal: usize },
}

impl Device {
	/// Select the best available device.
	///
	/// Probes in order: CUDA → HIP → wgpu → CPU.
	#[must_use]
	pub fn best() -> Self {
		#[cfg(feature = "cuda")]
		{
			// TODO: probe CUDA runtime
		}
		#[cfg(feature = "hip")]
		{
			if std::panic::catch_unwind(|| {
				crate::storage::hip_backend();
			})
			.is_ok()
			{
				return Self::Hip { ordinal: 0 };
			}
		}
		#[cfg(feature = "wgpu")]
		{
			// Try to init wgpu backend. If it succeeds, a GPU
			// is available. The OnceLock caches it for later.
			if std::panic::catch_unwind(|| {
				crate::storage::wgpu_backend();
			})
			.is_ok()
			{
				return Self::Wgpu { adapter: 0 };
			}
		}
		Self::Cpu
	}
}

impl std::fmt::Display for Device {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		match self {
			Self::Cpu => write!(f, "cpu"),
			#[cfg(feature = "wgpu")]
			Self::Wgpu { adapter } => {
				write!(f, "wgpu:{adapter}")
			}
			#[cfg(feature = "cuda")]
			Self::Cuda { ordinal } => {
				write!(f, "cuda:{ordinal}")
			}
			#[cfg(feature = "hip")]
			Self::Hip { ordinal } => {
				write!(f, "hip:{ordinal}")
			}
		}
	}
}
