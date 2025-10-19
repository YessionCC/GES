

/*%%HEADER%%*/
#define FULL_SH
/*%%DEFINES%%*/

uniform vec3 eye;

// spherical harmonics coeff for radiance of the splat
in vec4 r_sh0;  // sh coeff for red channel (up to third-order)
#ifdef FULL_SH
in vec4 r_sh1;
in vec4 r_sh2;
in vec4 r_sh3;
#endif
in vec4 g_sh0;  // sh coeff for green channel
#ifdef FULL_SH
in vec4 g_sh1;
in vec4 g_sh2;
in vec4 g_sh3;
#endif
in vec4 b_sh0;  // sh coeff for blue channel
#ifdef FULL_SH
in vec4 b_sh1;
in vec4 b_sh2;
in vec4 b_sh3;
#endif

// center of the gaussian in object coordinates, (with alpha crammed in to w)
in vec4 position;  
in vec3 splatRS_0;
in vec3 splatRS_1;

out vec4 geom_color;  // radiance of splat
out vec3 geom_splatRS0;
out vec3 geom_splatRS1;

vec3 ComputeRadianceFromSH(const vec3 v) {
#ifdef FULL_SH
    vec4 b0, b1, b2, b3;
#else
    vec4 b0;
#endif
    // zeroth order
    // (/ 1.0 (* 2.0 (sqrt pi)))
    b0.x = 0.28209479177387814f;

    // first order
    // (/ (sqrt 3.0) (* 2 (sqrt pi)))
    float k1 = 0.4886025119029199f;
    b0.y = -k1 * v.y;
    b0.z = k1 * v.z;
    b0.w = -k1 * v.x;

#ifdef FULL_SH
    float xx = v.x * v.x, yy = v.y * v.y, zz = v.z * v.z;
    float xy = v.x * v.y, xz = v.x * v.z, yz = v.y * v.z;

    // second order
    // (/ (sqrt 15.0) (* 2 (sqrt pi)))
    float k2 = 1.0925484305920792f;
    // (/ (sqrt 5.0) (* 4 (sqrt  pi)))
    float k3 = 0.31539156525252005f;
    // (/ (sqrt 15.0) (* 4 (sqrt pi)))
    float k4 = 0.5462742152960396f;
    b1.x = k2 * xy;
    b1.y = -k2 * yz;
    b1.z = k3 * (3.0f * zz - 1.0f);
    b1.w = -k2 * xz;
    b2.x = k4 * (xx - yy);

    // third order
    // (/ (* (sqrt 2) (sqrt 35)) (* 8 (sqrt pi)))
    float k5 = 0.5900435899266435f;
    // (/ (sqrt 105) (* 2 (sqrt pi)))
    float k6 = 2.8906114426405543f;
    // (/ (* (sqrt 2) (sqrt 21)) (* 8 (sqrt pi)))
    float k7 = 0.4570457994644658f;
    // (/ (sqrt 7) (* 4 (sqrt pi)))
    float k8 = 0.37317633259011546f;
    // (/ (sqrt 105) (* 4 (sqrt pi)))
    float k9 = 1.4453057213202771f;
    b2.y = -k5 * v.y * (3.0f * xx - yy);
    b2.z = k6 * xy * v.z;
    b2.w = -k7 * v.y * (5.0f * zz - 1.0f);
    b3.x = k8 * v.z * (5.0f * zz - 3.0f);
    b3.y = -k7 * v.x * (5.0f * zz - 1.0f);
    b3.z = k9 * v.z * (xx - yy);
    b3.w = -k5 * v.x * (xx - 3.0f * yy);

    float re = dot(b0, r_sh0) + dot(b1, r_sh1) + dot(b2, r_sh2) + dot(b3, r_sh3);

    float gr = dot(b0, g_sh0) + dot(b1, g_sh1) + dot(b2, g_sh2) + dot(b3, g_sh3);

    float bl = dot(b0, b_sh0) + dot(b1, b_sh1) + dot(b2, b_sh2) + dot(b3, b_sh3);
#else
    float re = dot(b0, r_sh0);
    float gr = dot(b0, g_sh0);
    float bl = dot(b0, b_sh0);
#endif
    return vec3(0.5f, 0.5f, 0.5f) + vec3(re, gr, bl);
}


void main(void)
{
    float alpha = position.w; // for geom, alpha is mod_depth; for tex, alpha is alpha

    // compute radiance from sh
    vec3 v = normalize(position.xyz - eye);
    geom_color = vec4(ComputeRadianceFromSH(v), alpha);
    geom_splatRS0 = splatRS_0;
    geom_splatRS1 = splatRS_1;

    gl_Position = vec4(position.xyz, 1.0f);
}
