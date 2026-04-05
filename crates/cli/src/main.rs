use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use fluxion_core::Grid;
use fluxion_runtime::{Device, DiffusionSolver, Field};

#[derive(Parser, Debug)]
#[command(name = "fluxion", about = "GPU PDE solver")]
struct Args {
	/// Grid size (N x N).
	#[arg(short, long, default_value_t = 256)]
	size: usize,

	/// Diffusion coefficient.
	#[arg(short, long, default_value_t = 0.01)]
	alpha: f64,

	/// Number of time steps.
	#[arg(long, default_value_t = 1000)]
	steps: usize,
}

fn main() -> Result<()> {
	tracing_subscriber::fmt()
		.with_env_filter(
			tracing_subscriber::EnvFilter::from_default_env(),
		)
		.init();

	let args = Args::parse();
	let n = args.size;
	let h = 1.0 / (n - 1) as f64;
	let device = Device::best();

	println!("fluxion — 2D heat diffusion");
	println!("  grid:   {n}x{n}  (h={h:.6})");
	println!("  alpha:  {}", args.alpha);
	println!("  device: {device}");

	let grid = Grid::square(n, h)?;

	// Gaussian initial condition centered at (0.5, 0.5).
	let cx = 0.5_f64;
	let cy = 0.5_f64;
	let sigma = 0.1_f64;
	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let dx = x - cx;
			let dy = y - cy;
			let r2 = dx.mul_add(dx, dy * dy);
			data[row * n + col] =
				(-r2 / (2.0 * sigma * sigma)).exp();
		}
	}

	let mut u = Field::from_f64(grid, &data, device)?;
	let mut solver = DiffusionSolver::new(
		grid,
		args.alpha,
		None,
		device,
	)?;

	println!("  dt:     {:.6e}", solver.dt());
	println!("  steps:  {}", args.steps);
	println!();

	let initial_norm = u.norm_l2()?;

	let t0 = Instant::now();
	solver.step_n(&mut u, args.steps)?;
	let elapsed = t0.elapsed();

	let final_norm = u.norm_l2()?;
	let final_data = u.to_vec_f64();
	let peak: f64 = final_data
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap_or(0.0);

	println!("results:");
	println!(
		"  sim time:     {:.6e} s",
		solver.sim_time()
	);
	println!("  wall time:    {elapsed:.3?}");
	println!("  initial norm: {initial_norm:.6}");
	println!("  final norm:   {final_norm:.6}");
	println!("  peak value:   {peak:.6}");
	println!(
		"  throughput:   {:.2} Mpoints/s",
		(args.steps as f64 * n as f64 * n as f64)
			/ elapsed.as_secs_f64()
			/ 1e6
	);

	Ok(())
}
