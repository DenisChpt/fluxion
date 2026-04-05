use criterion::{
	BenchmarkId, Criterion, Throughput, criterion_group,
	criterion_main,
};
use fluxion_core::{DType, Grid};
use fluxion_runtime::{Device, Field};

fn bench_axpy(c: &mut Criterion) {
	let device = Device::Cpu;
	let mut group = c.benchmark_group("axpy");

	for &n in &[64, 128, 256, 512, 1024] {
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let x =
			Field::zeros(grid, DType::F64, device).unwrap();
		let mut y =
			Field::zeros(grid, DType::F64, device).unwrap();

		group.throughput(Throughput::Elements(
			(n * n) as u64,
		));
		group.bench_with_input(
			BenchmarkId::new("cpu", n),
			&n,
			|b, _| {
				b.iter(|| {
					y.axpy(2.0, &x).unwrap();
				});
			},
		);
	}
	group.finish();
}

fn bench_norm_l2(c: &mut Criterion) {
	let device = Device::Cpu;
	let mut group = c.benchmark_group("norm_l2");

	for &n in &[64, 128, 256, 512, 1024] {
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let mut field =
			Field::zeros(grid, DType::F64, device).unwrap();
		field.fill(1.0).unwrap();

		group.throughput(Throughput::Elements(
			(n * n) as u64,
		));
		group.bench_with_input(
			BenchmarkId::new("cpu", n),
			&n,
			|b, _| {
				b.iter(|| {
					field.norm_l2().unwrap();
				});
			},
		);
	}
	group.finish();
}

fn bench_fill(c: &mut Criterion) {
	let device = Device::Cpu;
	let mut group = c.benchmark_group("fill");

	for &n in &[64, 128, 256, 512, 1024] {
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let mut field =
			Field::zeros(grid, DType::F64, device).unwrap();

		group.throughput(Throughput::Elements(
			(n * n) as u64,
		));
		group.bench_with_input(
			BenchmarkId::new("cpu", n),
			&n,
			|b, _| {
				b.iter(|| {
					field.fill(42.0).unwrap();
				});
			},
		);
	}
	group.finish();
}

criterion_group!(benches, bench_axpy, bench_norm_l2, bench_fill);
criterion_main!(benches);
