#include <hip/hip_runtime.h>
#include <rocprim/block/block_reduce.hpp>

// Optimised reductions: grid-stride loop + register accumulation
// + rocprim::block_reduce (warp shuffles / DPP per arch).
//
// Launch with a FIXED number of blocks (e.g. 32-64), not
// ceil(N/BLOCK). Each thread walks the entire array in strides
// of gridDim.x * BLOCK, accumulating in registers. One
// block_reduce at the end produces one partial per block.
// Host sums 32-64 partials (essentially free).

constexpr int BLOCK = 256;

// ── Sum of squares (for norm_l2) ────────────────────

extern "C" __global__ void reduce_sum_sq_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int len)
{
    using Reduce = rocprim::block_reduce<double, BLOCK>;
    __shared__ typename Reduce::storage_type storage;

    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int gid = blockIdx.x * BLOCK + tid;

    // Accumulate in register across grid-stride loop.
    double acc = 0.0;
    for (; gid < len; gid += stride) {
        double v = input[gid];
        acc = fma(v, v, acc);
    }

    Reduce().reduce(acc, acc, storage, rocprim::plus<double>());

    if (tid == 0) output[blockIdx.x] = acc;
}

// ── Sum ─────────────────────────────────────────────

extern "C" __global__ void reduce_sum_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int len)
{
    using Reduce = rocprim::block_reduce<double, BLOCK>;
    __shared__ typename Reduce::storage_type storage;

    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int gid = blockIdx.x * BLOCK + tid;

    double acc = 0.0;
    for (; gid < len; gid += stride) {
        acc += input[gid];
    }

    Reduce().reduce(acc, acc, storage, rocprim::plus<double>());

    if (tid == 0) output[blockIdx.x] = acc;
}

// ── Max ─────────────────────────────────────────────

extern "C" __global__ void reduce_max_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int len)
{
    using Reduce = rocprim::block_reduce<double, BLOCK>;
    __shared__ typename Reduce::storage_type storage;

    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int gid = blockIdx.x * BLOCK + tid;

    double acc = -1e308;
    for (; gid < len; gid += stride) {
        double v = input[gid];
        acc = fmax(acc, v);
    }

    Reduce().reduce(acc, acc, storage, rocprim::maximum<double>());

    if (tid == 0) output[blockIdx.x] = acc;
}

// ── Min ─────────────────────────────────────────────

extern "C" __global__ void reduce_min_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int len)
{
    using Reduce = rocprim::block_reduce<double, BLOCK>;
    __shared__ typename Reduce::storage_type storage;

    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int gid = blockIdx.x * BLOCK + tid;

    double acc = 1e308;
    for (; gid < len; gid += stride) {
        double v = input[gid];
        acc = fmin(acc, v);
    }

    Reduce().reduce(acc, acc, storage, rocprim::minimum<double>());

    if (tid == 0) output[blockIdx.x] = acc;
}

// ── Dual dot product ────────────────────────────────
// Computes dot(a,b) and dot(c,d) in a single pass.
// One kernel, one sync, one readback — halves overhead.

extern "C" __global__ void dot2_f64(
    const double* __restrict__ a,
    const double* __restrict__ b,
    const double* __restrict__ c,
    const double* __restrict__ d,
    double* __restrict__ output, // [2 * gridDim.x]
    int len)
{
    using Reduce = rocprim::block_reduce<double, BLOCK>;
    __shared__ typename Reduce::storage_type storage1;
    __shared__ typename Reduce::storage_type storage2;

    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int gid = blockIdx.x * BLOCK + tid;

    double acc_ab = 0.0;
    double acc_cd = 0.0;
    for (; gid < len; gid += stride) {
        acc_ab = fma(a[gid], b[gid], acc_ab);
        acc_cd = fma(c[gid], d[gid], acc_cd);
    }

    Reduce().reduce(acc_ab, acc_ab, storage1,
                    rocprim::plus<double>());
    // Second reduction needs a sync to reuse shared mem.
    __syncthreads();
    Reduce().reduce(acc_cd, acc_cd, storage2,
                    rocprim::plus<double>());

    if (tid == 0) {
        output[blockIdx.x] = acc_ab;
        output[gridDim.x + blockIdx.x] = acc_cd;
    }
}

// ── Dot product ─────────────────────────────────────

extern "C" __global__ void dot_f64(
    const double* __restrict__ x,
    const double* __restrict__ y,
    double* __restrict__ output,
    int len)
{
    using Reduce = rocprim::block_reduce<double, BLOCK>;
    __shared__ typename Reduce::storage_type storage;

    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int gid = blockIdx.x * BLOCK + tid;

    double acc = 0.0;
    for (; gid < len; gid += stride) {
        acc = fma(x[gid], y[gid], acc);
    }

    Reduce().reduce(acc, acc, storage, rocprim::plus<double>());

    if (tid == 0) output[blockIdx.x] = acc;
}
