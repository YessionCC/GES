
/*%%HEADER%%*/

uniform int sampleNum;

layout(location = 0) in vec4 frag_color_depth; 

layout(location = 2) in vec3 center_world;
layout(location = 3) in vec3 tangent_u;
layout(location = 4) in vec3 tangent_v;

out vec4 out_color;

uniform mat4 invProjMat;
uniform mat4 invViewMat;
uniform vec2 viewport_size;
uniform vec3 eye;


int inEllipse(vec2 offset) {
    vec2 sampleCoord = gl_FragCoord.xy + offset - vec2(0.5f);
    vec2 ndc = (sampleCoord / viewport_size) * 2.0 - 1.0;

    vec4 clip = vec4(ndc, -1.0, 1.0);

    vec4 view = invProjMat * clip;
    view /= view.w;

    vec3 dir = normalize((invViewMat * vec4(view.xyz, 0.0)).xyz);

    vec3 normal = normalize(cross(tangent_u, tangent_v));
    float t = dot(center_world - eye, normal) / dot(dir, normal);
    vec3 p_world = eye + t * dir;

    vec3 local = p_world - center_world;

    float u = dot(local, tangent_u) / dot(tangent_u, tangent_u);
    float v = dot(local, tangent_v) / dot(tangent_v, tangent_v);
    vec2 uv = vec2(u, v);

    float logG = -0.5f * dot(uv, uv);
    return (logG >= log(1.0f / 255.0f)) ? 1: 0;
}


void main()
{
    out_color = frag_color_depth;

    int sampleMask = 0;

    // The sample_positions are dependent on specified device
    if (sampleNum == 4) {
        sampleMask |= inEllipse(vec2(0.375f,0.125f));
        sampleMask |= (inEllipse(vec2(0.875f,0.375f)) << 1);
        sampleMask |= (inEllipse(vec2(0.125f,0.625f)) << 2);
        sampleMask |= (inEllipse(vec2(0.625f,0.875f)) << 3);
    }
    else if (sampleNum == 16) {
        sampleMask |= inEllipse(vec2(0.0625, 0.0000));
        sampleMask |= (inEllipse(vec2(0.2500, 0.1250)) << 1);
        sampleMask |= (inEllipse(vec2(0.1875, 0.3750)) << 2);
        sampleMask |= (inEllipse(vec2(0.4375, 0.3125)) << 3);
        sampleMask |= (inEllipse(vec2(0.5000, 0.0625)) << 4);
        sampleMask |= (inEllipse(vec2(0.6875, 0.1875)) << 5);
        sampleMask |= (inEllipse(vec2(0.7500, 0.4375)) << 6);
        sampleMask |= (inEllipse(vec2(0.9375, 0.2500)) << 7);
        sampleMask |= (inEllipse(vec2(0.0000, 0.5000)) << 8);
        sampleMask |= (inEllipse(vec2(0.3125, 0.6250)) << 9);
        sampleMask |= (inEllipse(vec2(0.1250, 0.7500)) << 10);
        sampleMask |= (inEllipse(vec2(0.3750, 0.8750)) << 11);
        sampleMask |= (inEllipse(vec2(0.5625, 0.5625)) << 12);
        sampleMask |= (inEllipse(vec2(0.8125, 0.6875)) << 13);
        sampleMask |= (inEllipse(vec2(0.6250, 0.8125)) << 14);
        sampleMask |= (inEllipse(vec2(0.8750, 0.9375)) << 15);
    }
    else {
        sampleMask = -1*inEllipse(vec2(0.0f, 0.0f));
        // no AA effect
    }
    gl_SampleMask[0] = sampleMask;

}
