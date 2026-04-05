use criterion::{
	BenchmarkId, Criterion, Throughput, criterion_group,
	criterion_main,
};
use fluxion_bench::quadratic_field;
use fluxion_core::{Boundaries, DType, Grid, Stencil};
use fluxion_runtime::{Device, Field};

fn bench_stencil_apply(c: &mut Criterion) {
	let device = Device::Cpu;
	let mut group = c.benchmark_group("stencil_laplacian_2d");

	for &n in &[64, 128, 256, 512, 1024] {
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let stencil = Stencil::laplacian_2d_5pt(h, h);
		let (_, field) = quadratic_field(n, device).unwrap();
		let mut output =
			Field::zeros(grid, DType::F64, device).unwrap();

		group.throughput(Throughput::Elements(
			(n * n) as u64,
		));
		group.bench_with_input(
			BenchmarkId::new("cpu", n),
			&n,
			|b, _| {
				b.iter(|| {
					field
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

criterion_group!(benches, bench_stencil_apply);
criterion_main!(benches);
