/*
    Copyright (c) 2024 Anthony J. Thibault
    This software is licensed under the MIT License. See LICENSE for more details.
*/

//
// 3d gaussian splat fragment shader
//

/*%%HEADER%%*/

in vec4 frag_color;  // radiance of splat
in vec4 frag_cov2inv;  // inverse of the 2D screen space covariance matrix of the guassian
in vec2 frag_p;  // 2D screen space center of the guassian

out vec4 out_color;


void main()
{
    vec2 d = gl_FragCoord.xy - frag_p;

    mat2 cov2Dinv = mat2(frag_cov2inv.xy, frag_cov2inv.zw);
    float g = exp(-0.5f * dot(d, cov2Dinv * d));

    float alpha = frag_color.a * g;

    out_color.rgb = alpha * frag_color.rgb;
    out_color.a = alpha;

    if (alpha < 1.0f / 255.0f)
    {
        discard;
    }
}
