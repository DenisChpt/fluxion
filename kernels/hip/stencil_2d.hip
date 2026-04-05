#include <hip/hip_runtime.h>

// 2D stencil kernels with LDS tiling.
// Block: 16x16 threads = 256 = 4 wavefronts (optimal for AMD).
// Halo loading distributed across ALL threads for max efficiency.

constexpr int TILE_X = 16;
constexpr int TILE_Y = 16;
// Total halo cells: 2*TILE_X + 2*TILE_Y = 64 edges
// (+ 4 corners for 9-point).
constexpr int HALO_EDGES = 2 * TILE_X + 2 * TILE_Y;

extern "C" __global__ void stencil_2d_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int rows, int cols,
    double inv_dx2, double inv_dy2)
{
    __shared__ double tile[TILE_Y + 2][TILE_X + 2];

    int col = blockIdx.x * TILE_X + threadIdx.x;
    int row = blockIdx.y * TILE_Y + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // Load center of tile (every thread loads one element).
    if (row < rows && col < cols)
        tile[ty][tx] = input[row * cols + col];
    else
        tile[ty][tx] = 0.0;

    // Distribute halo loading across all threads.
    // Linear thread ID in the block.
    int tid = threadIdx.y * TILE_X + threadIdx.x;
    if (tid < HALO_EDGES) {
        int hr, hc;  // halo row, col in tile coords
        int gr, gc;  // global row, col
        if (tid < TILE_X) {
            // Top halo: tile[0][1..TILE_X]
            hr = 0; hc = tid + 1;
            gr = (int)(blockIdx.y * TILE_Y) - 1;
            gc = (int)(blockIdx.x * TILE_X) + tid;
        } else if (tid < 2 * TILE_X) {
            // Bottom halo: tile[TILE_Y+1][1..TILE_X]
            hr = TILE_Y + 1; hc = (tid - TILE_X) + 1;
            gr = (int)(blockIdx.y * TILE_Y) + TILE_Y;
            gc = (int)(blockIdx.x * TILE_X) + (tid - TILE_X);
        } else if (tid < 2 * TILE_X + TILE_Y) {
            // Left halo: tile[1..TILE_Y][0]
            int k = tid - 2 * TILE_X;
            hr = k + 1; hc = 0;
            gr = (int)(blockIdx.y * TILE_Y) + k;
            gc = (int)(blockIdx.x * TILE_X) - 1;
        } else {
            // Right halo: tile[1..TILE_Y][TILE_X+1]
            int k = tid - 2 * TILE_X - TILE_Y;
            hr = k + 1; hc = TILE_X + 1;
            gr = (int)(blockIdx.y * TILE_Y) + k;
            gc = (int)(blockIdx.x * TILE_X) + TILE_X;
        }
        if (gr >= 0 && gr < rows && gc >= 0 && gc < cols)
            tile[hr][hc] = input[gr * cols + gc];
        else
            tile[hr][hc] = 0.0;
    }

    __syncthreads();

    // Boundary: Dirichlet zero.
    if (row == 0 || row >= rows - 1 || col == 0 || col >= cols - 1) {
        if (row < rows && col < cols)
            output[row * cols + col] = 0.0;
        return;
    }

    double center = -2.0 * (inv_dx2 + inv_dy2);
    double val = fma(center, tile[ty][tx],
                 fma(inv_dy2, tile[ty - 1][tx],
                 fma(inv_dy2, tile[ty + 1][tx],
                 fma(inv_dx2, tile[ty][tx - 1],
                     inv_dx2 * tile[ty][tx + 1]))));

    output[row * cols + col] = val;
}

// 9-point compact Laplacian with distributed halo loading.
extern "C" __global__ void stencil_9pt_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int rows, int cols,
    double w_c, double w_ns, double w_ew, double w_diag)
{
    __shared__ double tile[TILE_Y + 2][TILE_X + 2];

    int col = blockIdx.x * TILE_X + threadIdx.x;
    int row = blockIdx.y * TILE_Y + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // Load center.
    if (row < rows && col < cols)
        tile[ty][tx] = input[row * cols + col];
    else
        tile[ty][tx] = 0.0;

    // Distribute edge + corner halo across all threads.
    // 64 edges + 4 corners = 68 halo cells, 256 threads.
    int tid = threadIdx.y * TILE_X + threadIdx.x;
    if (tid < HALO_EDGES + 4) {
        int hr, hc, gr, gc;
        if (tid < TILE_X) {
            hr = 0; hc = tid + 1;
            gr = (int)(blockIdx.y * TILE_Y) - 1;
            gc = (int)(blockIdx.x * TILE_X) + tid;
        } else if (tid < 2 * TILE_X) {
            hr = TILE_Y + 1; hc = (tid - TILE_X) + 1;
            gr = (int)(blockIdx.y * TILE_Y) + TILE_Y;
            gc = (int)(blockIdx.x * TILE_X) + (tid - TILE_X);
        } else if (tid < 2 * TILE_X + TILE_Y) {
            int k = tid - 2 * TILE_X;
            hr = k + 1; hc = 0;
            gr = (int)(blockIdx.y * TILE_Y) + k;
            gc = (int)(blockIdx.x * TILE_X) - 1;
        } else if (tid < HALO_EDGES) {
            int k = tid - 2 * TILE_X - TILE_Y;
            hr = k + 1; hc = TILE_X + 1;
            gr = (int)(blockIdx.y * TILE_Y) + k;
            gc = (int)(blockIdx.x * TILE_X) + TILE_X;
        } else {
            // 4 corners (tid = 64, 65, 66, 67).
            int c = tid - HALO_EDGES;
            hr = (c < 2) ? 0 : TILE_Y + 1;
            hc = (c == 0 || c == 2) ? 0 : TILE_X + 1;
            gr = (c < 2)
                ? (int)(blockIdx.y * TILE_Y) - 1
                : (int)(blockIdx.y * TILE_Y) + TILE_Y;
            gc = (c == 0 || c == 2)
                ? (int)(blockIdx.x * TILE_X) - 1
                : (int)(blockIdx.x * TILE_X) + TILE_X;
        }
        if (gr >= 0 && gr < rows && gc >= 0 && gc < cols)
            tile[hr][hc] = input[gr * cols + gc];
        else
            tile[hr][hc] = 0.0;
    }

    __syncthreads();

    if (row == 0 || row >= rows - 1 || col == 0 || col >= cols - 1) {
        if (row < rows && col < cols)
            output[row * cols + col] = 0.0;
        return;
    }

    double val = fma(w_c, tile[ty][tx],
                 fma(w_ns, tile[ty-1][tx],
                 fma(w_ns, tile[ty+1][tx],
                 fma(w_ew, tile[ty][tx-1],
                 fma(w_ew, tile[ty][tx+1],
                 fma(w_diag, tile[ty-1][tx-1],
                 fma(w_diag, tile[ty-1][tx+1],
                 fma(w_diag, tile[ty+1][tx-1],
                     w_diag * tile[ty+1][tx+1]))))))));

    output[row * cols + col] = val;
}

extern "C" __global__ void stencil_2d_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols,
    float inv_dx2, float inv_dy2)
{
    __shared__ float tile[TILE_Y + 2][TILE_X + 2];

    int col = blockIdx.x * TILE_X + threadIdx.x;
    int row = blockIdx.y * TILE_Y + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    if (row < rows && col < cols)
        tile[ty][tx] = input[row * cols + col];
    else
        tile[ty][tx] = 0.0f;

    int tid = threadIdx.y * TILE_X + threadIdx.x;
    if (tid < HALO_EDGES) {
        int hr, hc, gr, gc;
        if (tid < TILE_X) {
            hr = 0; hc = tid + 1;
            gr = (int)(blockIdx.y * TILE_Y) - 1;
            gc = (int)(blockIdx.x * TILE_X) + tid;
        } else if (tid < 2 * TILE_X) {
            hr = TILE_Y + 1; hc = (tid - TILE_X) + 1;
            gr = (int)(blockIdx.y * TILE_Y) + TILE_Y;
            gc = (int)(blockIdx.x * TILE_X) + (tid - TILE_X);
        } else if (tid < 2 * TILE_X + TILE_Y) {
            int k = tid - 2 * TILE_X;
            hr = k + 1; hc = 0;
            gr = (int)(blockIdx.y * TILE_Y) + k;
            gc = (int)(blockIdx.x * TILE_X) - 1;
        } else {
            int k = tid - 2 * TILE_X - TILE_Y;
            hr = k + 1; hc = TILE_X + 1;
            gr = (int)(blockIdx.y * TILE_Y) + k;
            gc = (int)(blockIdx.x * TILE_X) + TILE_X;
        }
        if (gr >= 0 && gr < rows && gc >= 0 && gc < cols)
            tile[hr][hc] = input[gr * cols + gc];
        else
            tile[hr][hc] = 0.0f;
    }

    __syncthreads();

    if (row == 0 || row >= rows - 1 || col == 0 || col >= cols - 1) {
        if (row < rows && col < cols)
            output[row * cols + col] = 0.0f;
        return;
    }

    float center = -2.0f * (inv_dx2 + inv_dy2);
    float val = fmaf(center, tile[ty][tx],
                fmaf(inv_dy2, tile[ty - 1][tx],
                fmaf(inv_dy2, tile[ty + 1][tx],
                fmaf(inv_dx2, tile[ty][tx - 1],
                     inv_dx2 * tile[ty][tx + 1]))));

    output[row * cols + col] = val;
}
