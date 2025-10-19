

/*%%HEADER%%*/
/*%%DEFINES%%*/

out vec4 out_color;

in vec2 frag_uv;

uniform sampler2D colorWeight;

void main(void)
{
    ivec2 texCoord = ivec2(gl_FragCoord.xy);
    vec4 c_w = texelFetch(colorWeight, texCoord, 0);

    out_color = vec4(c_w.rgb/(c_w.a+1), 1.0);
}
