// Extended vector operations: pointwise_mult, pointwise_div,
// reciprocal, abs_val.
// Each is a separate entry point sharing the same layout.

struct Params {
    len: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn pointwise_mult(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) { return; }
    c[i] = a[i] * b[i];
}

@compute @workgroup_size(256)
fn pointwise_div(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) { return; }
    c[i] = a[i] / b[i];
}

// Reciprocal and abs use only binding 3 (in-place on c).

@compute @workgroup_size(256)
fn reciprocal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) { return; }
    c[i] = 1.0 / c[i];
}

@compute @workgroup_size(256)
fn abs_val(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) { return; }
    c[i] = abs(c[i]);
}
