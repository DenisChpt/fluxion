//! STREAM-like memory bandwidth benchmarks.
//!
//! Measures effective memory bandwidth (GB/s) for core
//! operations. Useful for roofline analysis — stencils are
//! memory-bound, so these numbers set the performance ceiling.

use criterion::{
	BenchmarkId, Criterion, Throughput, criterion_group,
	criterion_main,
};
use fluxion_core::{Boundaries, DType, Grid};
use fluxion_runtime::{Device, Field};

/// STREAM COPY equivalent: output = input.
/// Measures raw copy throughput.
fn bench_copy(c: &mut Criterion) {
	let device = Device::Cpu;
	let mut group = c.benchmark_group("bandwidth_copy");

	for &n in &[256, 512, 1024, 2048] {
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let data = vec![1.0_f64; n * n];
		let src =
			Field::from_f64(grid, &data, device).unwrap();

		// Bytes: read n² f64 + write n² f64 = 2 * n² * 8.
		group.throughput(Throughput::Bytes(
			(2 * n * n * 8) as u64,
		));
		group.bench_with_input(
			BenchmarkId::new("cpu", n),
			&n,
			|b, _| {
				b.iter_batched(
					|| {
						Field::zeros(
							grid,
							DType::F64,
							device,
						)
						.unwrap()
					},
					|mut dst| {
						dst.axpy(1.0, &src).unwrap();
					},
					criterion::BatchSize::LargeInput,
				);
			},
		);
	}
	group.finish();
}

/// STREAM TRIAD equivalent: a = b + α·c.
/// Most relevant for stencil-like operations.
fn bench_triad(c: &mut Criterion) {
	let device = Device::Cpu;
	let mut group = c.benchmark_group("bandwidth_triad");

	for &n in &[256, 512, 1024, 2048] {
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let data = vec![1.0_f64; n * n];
		let x =
			Field::from_f64(grid, &data, device).unwrap();
		let mut y =
			Field::from_f64(grid, &data, device).unwrap();

		// Bytes: read 2×n²×8 (x + y) + write n²×8 (y) = 3×n²×8.
		group.throughput(Throughput::Bytes(
			(3 * n * n * 8) as u64,
		));
		group.bench_with_input(
			BenchmarkId::new("cpu", n),
			&n,
			|b, _| {
				b.iter(|| {
					y.axpy(2.5, &x).unwrap();
				});
			},
		);
	}
	group.finish();
}

/// Stencil bandwidth: 5 reads + 1 write per point.
fn bench_stencil_bandwidth(c: &mut Criterion) {
	let device = Device::Cpu;
	let mut group =
		c.benchmark_group("bandwidth_stencil");

	for &n in &[256, 512, 1024, 2048] {
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let stencil =
			fluxion_core::Stencil::laplacian_2d_5pt(h, h);
		let data = vec![1.0_f64; n * n];
		let input = Field::from_f64(grid, &data, device)
			.unwrap();
		let mut output = Field::zeros(
			grid,
			DType::F64,
			device,
		)
		.unwrap();

		// 5 reads + 1 write per interior point, each 8 bytes.
		group.throughput(Throughput::Bytes(
			(6 * n * n * 8) as u64,
		));
		group.bench_with_input(
			BenchmarkId::new("cpu", n),
			&n,
			|b, _| {
				b.iter(|| {
					input
						.apply_stencil_into(
							&stencil,
							&Boundaries::zero_dirichlet(),
							&mut output,
						)
						.unwrap();
				});
			},
		);
	}
	group.finish();
}

criterion_group!(
	benches,
	bench_copy,
	bench_triad,
	bench_stencil_bandwidth
);
criterion_main!(benches);
