// Vertex shader



struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec3<f32>,
};

struct InstanceInput {
    @location(2) position: vec4<f32>
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) instance: f32,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.color = model.color;
    out.clip_position = vec4<f32>(
        model.position[0],
        model.position[1],
        0.0,
        1.0
    ) * instance.position;
    out.instance = instance.position[3];
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(
        in.color[0] * in.instance,
        in.color[1] * in.instance,
        in.color[2] * in.instance,
        1.0
    );
}
