

/*%%HEADER%%*/
/*%%DEFINES%%*/

out vec4 out_color;

in vec2 frag_uv;

uniform float zNear;
uniform float zFar;

uniform int sampleNum;
uniform sampler2DMS msColorDepth;

float depthSample(float linearDepth) {
    float nonLinearDepth = (zFar + zNear - 2.0 * zNear * zFar / linearDepth) / (zFar - zNear);
    nonLinearDepth = (nonLinearDepth + 1.0) / 2.0;
    return nonLinearDepth;
}


void main(void)
{
    ivec2 texCoord = ivec2(gl_FragCoord.xy);
    vec3 final_color = vec3(0.0);
    float final_depth = 0.0;
    for (int i = 0; i<sampleNum; i++) {
        vec4 color_depth = texelFetch(msColorDepth, texCoord, i);
        final_color += color_depth.xyz;
        // if color_depth.w==0, there is bg without geo gs, its linear depth should be infinity far 
        float depth = color_depth.w == 0 ? -zFar : color_depth.w;
        final_depth += depth;
    }
    final_color /= sampleNum;
    final_depth /= sampleNum;
    final_depth = depthSample(-final_depth);

    out_color = vec4(final_color, 0.0);
    //out_color = vec4(final_depth, final_depth, final_depth, 0.0);
    gl_FragDepth = final_depth;
}
