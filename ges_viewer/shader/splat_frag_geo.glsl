
/*%%HEADER%%*/

uniform int sampleNum;

layout(location = 0) in vec4 frag_color_depth;  // NOTE: depth is linear
layout(location = 1) in vec2 frag_uv;

out vec4 out_color;

int inEllipse(vec2 offset, vec4 duvdxy) {
    offset -= vec2(0.5);
    vec2 uv = frag_uv + offset.x * duvdxy.xy + offset.y * duvdxy.zw;
    float logG = -0.5f * dot(uv, uv);
    return (logG >= log(1.0f / 255.0f)) ? 1: 0;
}

void main()
{
    vec4 duvdxy = vec4(dFdx(frag_uv), dFdy(frag_uv));

    out_color = frag_color_depth;
    
    int sampleMask = 0;

    if (sampleNum == 4) {
        sampleMask |= inEllipse(vec2(0.375f,0.125f), duvdxy);
        sampleMask |= (inEllipse(vec2(0.875f,0.375f), duvdxy) << 1);
        sampleMask |= (inEllipse(vec2(0.125f,0.625f), duvdxy) << 2);
        sampleMask |= (inEllipse(vec2(0.625f,0.875f), duvdxy) << 3);
    }
    else if (sampleNum == 16) {
        sampleMask |= inEllipse(vec2(0.0625, 0.0000), duvdxy);
        sampleMask |= (inEllipse(vec2(0.2500, 0.1250), duvdxy) << 1);
        sampleMask |= (inEllipse(vec2(0.1875, 0.3750), duvdxy) << 2);
        sampleMask |= (inEllipse(vec2(0.4375, 0.3125), duvdxy) << 3);
        sampleMask |= (inEllipse(vec2(0.5000, 0.0625), duvdxy) << 4);
        sampleMask |= (inEllipse(vec2(0.6875, 0.1875), duvdxy) << 5);
        sampleMask |= (inEllipse(vec2(0.7500, 0.4375), duvdxy) << 6);
        sampleMask |= (inEllipse(vec2(0.9375, 0.2500), duvdxy) << 7);
        sampleMask |= (inEllipse(vec2(0.0000, 0.5000), duvdxy) << 8);
        sampleMask |= (inEllipse(vec2(0.3125, 0.6250), duvdxy) << 9);
        sampleMask |= (inEllipse(vec2(0.1250, 0.7500), duvdxy) << 10);
        sampleMask |= (inEllipse(vec2(0.3750, 0.8750), duvdxy) << 11);
        sampleMask |= (inEllipse(vec2(0.5625, 0.5625), duvdxy) << 12);
        sampleMask |= (inEllipse(vec2(0.8125, 0.6875), duvdxy) << 13);
        sampleMask |= (inEllipse(vec2(0.6250, 0.8125), duvdxy) << 14);
        sampleMask |= (inEllipse(vec2(0.8750, 0.9375), duvdxy) << 15);
    }
    else {
        sampleMask = -1*inEllipse(vec2(0.0f, 0.0f), duvdxy);
        // no AA effect
    }
    gl_SampleMask[0] = sampleMask;
    

}
