
/*%%HEADER%%*/

in vec4 frag_color;  // radiance of splat
in vec2 frag_uv;

out vec4 out_color;


void main()
{
    float g = exp(-0.5f * dot(frag_uv, frag_uv));

    float alpha = frag_color.a * g;

    out_color.rgb = alpha * frag_color.rgb;
    out_color.a = alpha;

    if (alpha < 1.0f / 255.0f)
    {
        discard;
    }
}
