// Parallel reduction: compute partial min per workgroup.
// Output: one f32 per workgroup.

struct Params {
    len: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared: array<f32, 256>;

const POS_INF: f32 = 3.402823466e+38;

@compute
@workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let gid = wid.x * 256u + tid;

    if (gid < params.len) {
        shared[tid] = input[gid];
    } else {
        shared[tid] = POS_INF;
    }
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (tid < stride) {
            shared[tid] = min(shared[tid], shared[tid + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (tid == 0u) {
        output[wid.x] = shared[0];
    }
}
