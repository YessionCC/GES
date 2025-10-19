/*
    Copyright (c) 2024 Anthony J. Thibault
    This software is licensed under the MIT License. See LICENSE for more details.
*/

/*%%HEADER%%*/

uniform vec4 viewport;  // x, y, WIDTH, HEIGHT

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec4 geom_color[];  // radiance of splat
in vec4 geom_cov2[];  // 2D screen space covariance matrix of the gaussian
in vec2 geom_p[];  // the 2D screen space center of the gaussian
in float geom_pviewz[];

out vec4 frag_color;  // radiance of splat
out vec4 frag_cov2inv;  // inverse of the 2D screen space covariance matrix of the guassian
out vec2 frag_p;  // the 2D screen space center of the gaussian


void main()
{
    float WIDTH = viewport.z;
    float HEIGHT = viewport.w;

    mat2 cov2D = mat2(geom_cov2[0].xy, geom_cov2[0].zw);
    // we pass the inverse of the 2d covariance matrix to the pixel shader, to avoid doing a matrix inverse per pixel.
    float det = cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[1][0];
    if (det == 0) return;
    mat2 cov2Dinv;
    cov2Dinv[0][0] =  cov2D[1][1] / det;
    cov2Dinv[0][1] = -cov2D[0][1] / det;
    cov2Dinv[1][0] = -cov2D[1][0] / det;
    cov2Dinv[1][1] =  cov2D[0][0] / det;

    vec4 cov2Dinv4 = vec4(cov2Dinv[0], cov2Dinv[1]); // cram it into a vec4

    // discard splats that end up outside of a guard band
    if (geom_pviewz[0] > -0.2f)
    {
        // discard this point
        return;
    }

    // compute 2d extents for the splat, using covariance matrix ellipse
    // see https://cookierobotics.com/007/
    float k = 3.4f;
    float a = cov2D[0][0];
    float b = cov2D[0][1];
    float c = cov2D[1][1];
    float apco2 = (a + c) / 2.0f;
    float amco2 = (a - c) / 2.0f;
    float term = sqrt(amco2 * amco2 + b * b);
    float majv = apco2 + term;
    float minv = apco2 - term;

    float r1 = k * sqrt(majv);
    float r2 = k * sqrt(minv);

    vec2 majAxis, minAxis;
    if (b == 0.0f) {
        if (a >= c) {
            majAxis = vec2(r1, 0.0f);
            minAxis = vec2(0.0f, r2);
        }
        else {
            majAxis = vec2(0.0f, r1);
            minAxis = vec2(-r2, 0.0f);
        }
    }
    else {
        vec2 axis_dir = vec2(b, majv-a);
        axis_dir = normalize(axis_dir);
        majAxis = r1*axis_dir;
        minAxis = r2*vec2(-axis_dir.y, axis_dir.x);
    }

    vec2 offsets[4];
    offsets[0] = majAxis + minAxis;
    offsets[1] = -majAxis + minAxis;
    offsets[3] = -majAxis - minAxis;
    offsets[2] = majAxis - minAxis;


    vec2 offset;
    float w = gl_in[0].gl_Position.w;
    for (int i = 0; i < 4; i++)
    {
        // transform offset back into clip space, and apply it to gl_Position.
        offset = offsets[i];
        offset.x *= (2.0f / WIDTH) * w;
        offset.y *= (2.0f / HEIGHT) * w;

        gl_Position = gl_in[0].gl_Position + vec4(offset.x, offset.y, 0.0, 0.0);
        frag_color = geom_color[0];
        frag_cov2inv = cov2Dinv4;
        frag_p = geom_p[0];

        EmitVertex();
    }

    EndPrimitive();
}
