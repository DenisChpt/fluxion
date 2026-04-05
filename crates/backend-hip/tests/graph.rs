//! hipGraph capture and replay test.

#![allow(unsafe_code)]

use fluxion_core::{Backend, BackendBuffer, DType};
use fluxion_backend_hip::{HipBackend, HipGraph};

#[test]
fn graph_capture_and_replay() {
	let backend = HipBackend::new().unwrap();
	let stream = backend.stream();

	let mut buf =
		backend.allocate(256, DType::F64).unwrap();

	// Capture a fill + scale sequence.
	unsafe { HipGraph::begin_capture(stream) }.unwrap();
	backend.fill(&mut buf, 10.0).unwrap();
	backend.scale(&mut buf, 0.5).unwrap();
	let graph =
		unsafe { HipGraph::end_capture(stream) }.unwrap();

	// Reset buffer, then replay the graph.
	backend.fill(&mut buf, 0.0).unwrap();
	unsafe { graph.launch(stream) }.unwrap();
	unsafe {
		fluxion_backend_hip::ffi::hipStreamSynchronize(
			stream,
		);
	}

	let mut data = vec![0.0_f64; 256];
	buf.copy_to_host_f64(&mut data);
	assert!(
		data.iter().all(|&v| (v - 5.0).abs() < 1e-12),
		"graph replay: expected 5.0, got {}",
		data[0]
	);
}

#[test]
fn graph_replay_multiple_times() {
	let backend = HipBackend::new().unwrap();
	let stream = backend.stream();

	let mut buf =
		backend.allocate(1024, DType::F64).unwrap();

	// Capture: scale by 2.
	backend.fill(&mut buf, 1.0).unwrap();
	unsafe { HipGraph::begin_capture(stream) }.unwrap();
	backend.scale(&mut buf, 2.0).unwrap();
	let graph =
		unsafe { HipGraph::end_capture(stream) }.unwrap();

	// Replay 10 times: 1 * 2^10 = 1024.
	for _ in 0..10 {
		unsafe { graph.launch(stream) }.unwrap();
	}
	unsafe {
		fluxion_backend_hip::ffi::hipStreamSynchronize(
			stream,
		);
	}

	let mut data = vec![0.0_f64; 1024];
	buf.copy_to_host_f64(&mut data);
	assert!(
		(data[0] - 1024.0).abs() < 1e-8,
		"expected 1024.0, got {}",
		data[0]
	);
}
