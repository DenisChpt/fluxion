use criterion::{
	BenchmarkId, Criterion, Throughput, criterion_group,
	criterion_main,
};
use fluxion_bench::gaussian_field;
use fluxion_runtime::{Device, DiffusionSolver};

const STEPS: usize = 100;

fn bench_diffusion_solver(c: &mut Criterion) {
	let device = Device::Cpu;
	let alpha = 0.01;
	let mut group = c.benchmark_group("diffusion_euler");

	for &n in &[64, 128, 256, 512] {
		let (grid, _) = gaussian_field(n, device).unwrap();

		group.throughput(Throughput::Elements(
			(n as u64 * n as u64) * STEPS as u64,
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
							.step_n(&mut field, STEPS)
							.unwrap();
					},
					criterion::BatchSize::LargeInput,
				);
			},
		);
	}
	group.finish();
}

criterion_group!(benches, bench_diffusion_solver);
criterion_main!(benches);
