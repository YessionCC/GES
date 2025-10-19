

/*%%HEADER%%*/
/*%%DEFINES%%*/

out vec4 out_color;

uniform ivec2 texSize;

uniform sampler2D colorMap;

void main(void)
{
    ivec2 texCoord = ivec2(gl_FragCoord.xy);
    vec4 sum = vec4(0.0);

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            ivec2 sampleCoord = clamp(texCoord + ivec2(dx, dy), ivec2(0), texSize - ivec2(1));
            sum += texelFetch(colorMap, sampleCoord, 0);
        }
    }
    out_color = sum / 9.0;
}
