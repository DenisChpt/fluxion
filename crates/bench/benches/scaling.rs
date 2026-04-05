//! Grid size scaling benchmarks.
//!
//! Measures how performance scales with problem size.
//! For a memory-bound operation, throughput (points/s) should
//! plateau once the data exceeds cache size.

use criterion::{
	BenchmarkId, Criterion, Throughput, criterion_group,
	criterion_main,
};
use fluxion_bench::gaussian_field;
use fluxion_core::{Boundaries, DType, Grid, Stencil};
use fluxion_runtime::{Device, DiffusionSolver, Field};

/// Stencil scaling from 16² to 2048² (fits L1 → exceeds L3).
fn bench_stencil_scaling(c: &mut Criterion) {
	let device = Device::Cpu;
	let mut group =
		c.benchmark_group("scaling_stencil");
	group.sample_size(30);

	for &n in &[16, 32, 64, 128, 256, 512, 1024, 2048] {
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let stencil = Stencil::laplacian_2d_5pt(h, h);
		let data = vec![1.0_f64; n * n];
		let input = Field::from_f64(grid, &data, device)
			.unwrap();
		let mut output = Field::zeros(
			grid,
			DType::F64,
			device,
		)
		.unwrap();

		group.throughput(Throughput::Elements(
			(n * n) as u64,
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

/// Full solver scaling: 100 steps at various grid sizes.
fn bench_solver_scaling(c: &mut Criterion) {
	let device = Device::Cpu;
	let alpha = 0.01;
	let steps = 100;
	let mut group =
		c.benchmark_group("scaling_solver");
	group.sample_size(10);

	for &n in &[32, 64, 128, 256, 512] {
		let (grid, _) =
			gaussian_field(n, device).unwrap();

		group.throughput(Throughput::Elements(
			(n as u64 * n as u64) * steps as u64,
		));
		group.bench_with_input(
			BenchmarkId::new("cpu", n),
			&n,
			|b, _| {
				b.iter_batched(
					|| {
						let (_, field) =
							gaussian_field(n, device)
								.unwrap();
						let solver =
							DiffusionSolver::new(
								grid, alpha, None, device,
							)
							.unwrap();
						(field, solver)
					},
					|(mut field, mut solver)| {
						solver
							.step_n(&mut field, steps)
							.unwrap();
					},
					criterion::BatchSize::LargeInput,
				);
			},
		);
	}
	group.finish();
}

criterion_group!(
	benches,
	bench_stencil_scaling,
	bench_solver_scaling
);
criterion_main!(benches);
