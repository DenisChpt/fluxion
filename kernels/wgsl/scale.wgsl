struct Params {
    alpha: f32,
    len: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) {
        return;
    }
    buf[i] = params.alpha * buf[i];
}
