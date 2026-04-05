#include <hip/hip_runtime.h>

// Convection-diffusion operator with variable coefficients:
//   L(u) = div(kappa * grad(u)) + v · grad(u)
//
// Discretised on structured 2D grid with upwind for convection:
//   diffusion:  kappa[i,j] * laplacian(u)  (5-point stencil)
//   convection: vx * du/dx + vy * du/dy    (first-order upwind)
//
// kappa, vx, vy are per-cell fields (same grid as u).

constexpr int TILE_X = 16;
constexpr int TILE_Y = 16;

// ── Fused convection-diffusion + axpy ───────────────
// output += alpha * (kappa * Lap(u) + v · grad(u))
// LDS-tiled for u; kappa, vx, vy read from global (low reuse).

extern "C" __global__ void conv_diff_axpy_f64(
    double alpha,
    const double* __restrict__ u,
    const double* __restrict__ kappa,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    double* __restrict__ output,
    int rows, int cols,
    double inv_dx, double inv_dy,
    double inv_dx2, double inv_dy2)
{
    __shared__ double tile[TILE_Y + 2][TILE_X + 2];

    int col = blockIdx.x * TILE_X + threadIdx.x;
    int row = blockIdx.y * TILE_Y + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // Load u tile + halo.
    if (row < rows && col < cols)
        tile[ty][tx] = u[row * cols + col];
    else
        tile[ty][tx] = 0.0;

    if (threadIdx.y == 0 && row > 0)
        tile[0][tx] = u[(row - 1) * cols + col];
    if (threadIdx.y == TILE_Y - 1 && row + 1 < rows)
        tile[TILE_Y + 1][tx] = u[(row + 1) * cols + col];
    if (threadIdx.x == 0 && col > 0)
        tile[ty][0] = u[row * cols + (col - 1)];
    if (threadIdx.x == TILE_X - 1 && col + 1 < cols)
        tile[ty][TILE_X + 1] = u[row * cols + (col + 1)];

    __syncthreads();

    // Interior only.
    if (row <= 0 || row >= rows - 1 || col <= 0 || col >= cols - 1)
        return;

    int idx = row * cols + col;

    // Diffusion: kappa * laplacian(u).
    double center = -2.0 * (inv_dx2 + inv_dy2);
    double lap = fma(center, tile[ty][tx],
                 fma(inv_dy2, tile[ty - 1][tx],
                 fma(inv_dy2, tile[ty + 1][tx],
                 fma(inv_dx2, tile[ty][tx - 1],
                     inv_dx2 * tile[ty][tx + 1]))));
    double diff = kappa[idx] * lap;

    // Convection: upwind scheme.
    double vx_i = vx[idx];
    double vy_i = vy[idx];

    double dudx, dudy;
    // Upwind in x.
    if (vx_i >= 0.0)
        dudx = (tile[ty][tx] - tile[ty][tx - 1]) * inv_dx;
    else
        dudx = (tile[ty][tx + 1] - tile[ty][tx]) * inv_dx;
    // Upwind in y.
    if (vy_i >= 0.0)
        dudy = (tile[ty][tx] - tile[ty - 1][tx]) * inv_dy;
    else
        dudy = (tile[ty + 1][tx] - tile[ty][tx]) * inv_dy;

    double conv = fma(vx_i, dudx, vy_i * dudy);

    output[idx] = fma(alpha, diff + conv, output[idx]);
}

// ── Standalone convection-diffusion (non-fused) ────
// output = kappa * Lap(u) + v · grad(u)

extern "C" __global__ void conv_diff_f64(
    const double* __restrict__ u,
    const double* __restrict__ kappa,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    double* __restrict__ output,
    int rows, int cols,
    double inv_dx, double inv_dy,
    double inv_dx2, double inv_dy2)
{
    __shared__ double tile[TILE_Y + 2][TILE_X + 2];

    int col = blockIdx.x * TILE_X + threadIdx.x;
    int row = blockIdx.y * TILE_Y + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    if (row < rows && col < cols)
        tile[ty][tx] = u[row * cols + col];
    else
        tile[ty][tx] = 0.0;

    if (threadIdx.y == 0 && row > 0)
        tile[0][tx] = u[(row - 1) * cols + col];
    if (threadIdx.y == TILE_Y - 1 && row + 1 < rows)
        tile[TILE_Y + 1][tx] = u[(row + 1) * cols + col];
    if (threadIdx.x == 0 && col > 0)
        tile[ty][0] = u[row * cols + (col - 1)];
    if (threadIdx.x == TILE_X - 1 && col + 1 < cols)
        tile[ty][TILE_X + 1] = u[row * cols + (col + 1)];

    __syncthreads();

    if (row <= 0 || row >= rows - 1 || col <= 0 || col >= cols - 1) {
        if (row < rows && col < cols)
            output[row * cols + col] = 0.0;
        return;
    }

    int idx = row * cols + col;

    double center = -2.0 * (inv_dx2 + inv_dy2);
    double lap = fma(center, tile[ty][tx],
                 fma(inv_dy2, tile[ty - 1][tx],
                 fma(inv_dy2, tile[ty + 1][tx],
                 fma(inv_dx2, tile[ty][tx - 1],
                     inv_dx2 * tile[ty][tx + 1]))));
    double diff = kappa[idx] * lap;

    double vx_i = vx[idx];
    double vy_i = vy[idx];

    double dudx, dudy;
    if (vx_i >= 0.0)
        dudx = (tile[ty][tx] - tile[ty][tx - 1]) * inv_dx;
    else
        dudx = (tile[ty][tx + 1] - tile[ty][tx]) * inv_dx;
    if (vy_i >= 0.0)
        dudy = (tile[ty][tx] - tile[ty - 1][tx]) * inv_dy;
    else
        dudy = (tile[ty + 1][tx] - tile[ty][tx]) * inv_dy;

    output[idx] = fma(vx_i, dudx, vy_i * dudy) + diff;
}

// ── Naive stencil (no LDS, no tiling) for comparison ─
// Same 5-point Laplacian but direct global memory access.

extern "C" __global__ void stencil_naive_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int rows, int cols,
    double inv_dx2, double inv_dy2)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols) return;

    if (row == 0 || row >= rows - 1 || col == 0 || col >= cols - 1) {
        output[row * cols + col] = 0.0;
        return;
    }

    int idx = row * cols + col;
    double center = -2.0 * (inv_dx2 + inv_dy2);

    // Direct global memory reads — no shared memory.
    double val = fma(center, input[idx],
                 fma(inv_dy2, input[(row-1)*cols + col],
                 fma(inv_dy2, input[(row+1)*cols + col],
                 fma(inv_dx2, input[row*cols + (col-1)],
                     inv_dx2 * input[row*cols + (col+1)]))));

    output[idx] = val;
}
