#include <hip/hip_runtime.h>

// Element-wise operations: fill, scale, axpy.
// One element per thread, linear launch.
// For bandwidth-bound ops, the GPU memory coalescer already
// combines adjacent thread loads into wide transactions —
// vectorized types (double2) add instruction overhead without
// improving throughput.

constexpr int BLOCK = 256;

extern "C" __global__ void fill_f64(
    double* __restrict__ buf, double value, int len)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < len) buf[i] = value;
}

extern "C" __global__ void scale_f64(
    double* __restrict__ buf, double alpha, int len)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < len) buf[i] *= alpha;
}

extern "C" __global__ void axpy_f64(
    double alpha,
    const double* __restrict__ x,
    double* __restrict__ y,
    int len)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < len) y[i] = fma(alpha, x[i], y[i]);
}

// ── Pointwise multiply: z = x * y ──────────────────

extern "C" __global__ void pointwise_mult_f64(
    const double* __restrict__ x,
    const double* __restrict__ y,
    double* __restrict__ z,
    int len)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < len) z[i] = x[i] * y[i];
}

// ── Pointwise divide: z = x / y ────────────────────

extern "C" __global__ void pointwise_div_f64(
    const double* __restrict__ x,
    const double* __restrict__ y,
    double* __restrict__ z,
    int len)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < len) z[i] = x[i] / y[i];
}

// ── WAXPY: w = alpha*x + beta*y ────────────────────

extern "C" __global__ void waxpy_f64(
    double alpha,
    const double* __restrict__ x,
    double beta,
    const double* __restrict__ y,
    double* __restrict__ w,
    int len)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < len) w[i] = fma(alpha, x[i], beta * y[i]);
}

// ── AYPX: y = x + alpha*y ──────────────────────────

extern "C" __global__ void aypx_f64(
    double alpha,
    const double* __restrict__ x,
    double* __restrict__ y,
    int len)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < len) y[i] = fma(alpha, y[i], x[i]);
}

// ── Reciprocal: x = 1/x ────────────────────────────

extern "C" __global__ void reciprocal_f64(
    double* __restrict__ buf, int len)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < len) buf[i] = 1.0 / buf[i];
}

// ── Absolute value: x = |x| ────────────────────────

extern "C" __global__ void abs_f64(
    double* __restrict__ buf, int len)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < len) buf[i] = fabs(buf[i]);
}

// ── Pipelined CG fused 6-way update ────────────────
// Single pass: z,t,p,x,r,w all updated together.
// Eliminates 5 separate kernel launches.
//
// z = q + beta*z
// t = w + beta*t
// p = r + beta*p
// x += alpha*p   (uses UPDATED p)
// r -= alpha*t   (uses UPDATED t)
// w -= alpha*z   (uses UPDATED z)

extern "C" __global__ void pipelined_cg_fused_f64(
    double alpha, double beta,
    double* __restrict__ z,
    double* __restrict__ t,
    double* __restrict__ p,
    double* __restrict__ x,
    double* __restrict__ r,
    double* __restrict__ w,
    const double* __restrict__ q,
    int len)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < len) {
        double zi = fma(beta, z[i], q[i]);
        double ti = fma(beta, t[i], w[i]);
        double pi = fma(beta, p[i], r[i]);
        x[i] = fma(alpha, pi, x[i]);
        r[i] = fma(-alpha, ti, r[i]);
        w[i] = fma(-alpha, zi, w[i]);
        z[i] = zi;
        t[i] = ti;
        p[i] = pi;
    }
}

extern "C" __global__ void pointwise_max_f64(
    const double* __restrict__ x,
    const double* __restrict__ y,
    double* __restrict__ z, int len)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < len) z[i] = fmax(x[i], y[i]);
}

extern "C" __global__ void pointwise_min_f64(
    const double* __restrict__ x,
    const double* __restrict__ y,
    double* __restrict__ z, int len)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < len) z[i] = fmin(x[i], y[i]);
}

// Zero boundary rows/cols for Dirichlet BCs.
// One thread per element, only writes boundaries.
extern "C" __global__ void zero_boundaries_f64(
    double* __restrict__ buf, int rows, int cols)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    int len = rows * cols;
    if (i >= len) return;
    int row = i / cols;
    int col = i % cols;
    if (row == 0 || row == rows - 1
        || col == 0 || col == cols - 1)
    {
        buf[i] = 0.0;
    }
}
