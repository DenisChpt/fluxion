use std::ptr;

use crate::ffi;

/// RAII wrapper for a captured and instantiated hipGraph.
///
/// Records a sequence of HIP operations (kernel launches,
/// memcpy, etc.) and replays them with a single
/// `hipGraphLaunch`, eliminating per-launch overhead.
///
/// Typical speedup: 1.5-3x on small grids where launch
/// latency dominates.
#[derive(Debug)]
pub struct HipGraph {
	graph: ffi::hipGraph_t,
	exec: ffi::hipGraphExec_t,
}

// SAFETY: hipGraph handles are thread-safe via stream ordering.
unsafe impl Send for HipGraph {}
unsafe impl Sync for HipGraph {}

impl HipGraph {
	/// Begin capturing operations on the given stream.
	///
	/// All subsequent kernel launches on this stream will be
	/// recorded into the graph instead of executing immediately.
	/// Call `end_capture` to finalize.
	///
	/// # Safety
	/// `stream` must be a valid, non-null HIP stream handle.
	///
	/// # Errors
	/// Returns an error if capture fails.
	pub unsafe fn begin_capture(
		stream: ffi::hipStream_t,
	) -> Result<(), String> {
		ffi::check(unsafe {
			ffi::hipStreamBeginCapture(
				stream,
				ffi::hipStreamCaptureModeRelaxed,
			)
		})
	}

	/// End capture and instantiate the graph for replay.
	///
	/// # Safety
	/// `stream` must be a valid, non-null HIP stream handle
	/// that is currently in capture mode via `begin_capture`.
	///
	/// # Errors
	/// Returns an error if capture or instantiation fails.
	pub unsafe fn end_capture(
		stream: ffi::hipStream_t,
	) -> Result<Self, String> {
		let mut graph = ptr::null_mut();
		ffi::check(unsafe {
			ffi::hipStreamEndCapture(stream, &mut graph)
		})?;

		let mut exec = ptr::null_mut();
		ffi::check(unsafe {
			ffi::hipGraphInstantiate(
				&mut exec,
				graph,
				ptr::null_mut(),
				ptr::null_mut(),
				0,
			)
		})?;

		Ok(Self { graph, exec })
	}

	/// Replay the captured graph on the given stream.
	///
	/// All operations recorded during capture are replayed
	/// with minimal launch overhead.
	///
	/// # Safety
	/// `stream` must be a valid, non-null HIP stream handle.
	///
	/// # Errors
	/// Returns an error if launch fails.
	#[inline]
	pub unsafe fn launch(
		&self,
		stream: ffi::hipStream_t,
	) -> Result<(), String> {
		ffi::check(unsafe {
			ffi::hipGraphLaunch(self.exec, stream)
		})
	}
}

impl Drop for HipGraph {
	fn drop(&mut self) {
		unsafe {
			if !self.exec.is_null() {
				ffi::hipGraphExecDestroy(self.exec);
			}
			if !self.graph.is_null() {
				ffi::hipGraphDestroy(self.graph);
			}
		}
	}
}
