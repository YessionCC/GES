
/*%%HEADER%%*/

uniform mat4 viewMat;  // used to project position into view coordinates.
uniform mat4 projMat;  // used to project view coordinates into clip coordinates.
uniform vec2 fxfy; // focal length
uniform bool use_filter;
uniform float opac_thr;

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec4 geom_color[];  // radiance of splat
in vec3 geom_splatRS0[]; 
in vec3 geom_splatRS1[]; 

out vec4 frag_color; 
out vec2 frag_uv;


void main()
{
    mat4 pvMat = projMat * viewMat;
    // discard splats that end up outside of a guard band
    vec4 c_pos = gl_in[0].gl_Position;

    vec4 p_view = viewMat * c_pos;
    if (p_view.z > -0.2) {
        return;
    }
    
    /*
    vec4 p4 = pvMat * c_pos;
    vec3 ndcP = p4.xyz / p4.w;
    if (ndcP.z < 0.2f ||
        ndcP.x > 1.5f || ndcP.x < -1.5f ||
        ndcP.y > 1.5f || ndcP.y < -1.5f)
    {
        // discard this point
        return;
    }
    */
    
    
    //float cutoff = 3.4f; // 3.3
    
    float opacity = geom_color[0].w;
    float cutoff = sqrt(max(11.1f + 2.f * log(opacity), 0.000001));
    //cutoff = min(3.4f, cutoff);
    if (opacity > opac_thr) return;

    ////////////// For object space ewa filter
    float s_coeff1 = 1.0, s_coeff2 = 1.0, opac_coeff = 1.0;
    if (use_filter) {
        vec3 splatRS0_view = mat3(viewMat) * geom_splatRS0[0];
        vec3 splatRS1_view = mat3(viewMat) * geom_splatRS1[0];
        float fxdz = fxfy.x / (p_view.z * p_view.z);
        float fydz = fxfy.y / (p_view.z * p_view.z);
        float a = fxdz*(splatRS0_view.x*p_view.z - splatRS0_view.z*p_view.x);
        float b = fxdz*(splatRS1_view.x*p_view.z - splatRS1_view.z*p_view.x);
        float c = fydz*(splatRS0_view.z*p_view.y - splatRS0_view.y*p_view.z);
        float d = fydz*(splatRS1_view.z*p_view.y - splatRS1_view.y*p_view.z);

        float det_J = a*d - b*c, abs_det_J = abs(det_J), square_det_J = det_J * det_J;
        float t1 = d*d + b*b, t2 = a*a + c*c;

        if (abs_det_J < 1e-6) return; // nearly orthogonal to the screen

        s_coeff1 = sqrt(square_det_J + 0.3*t1) / abs_det_J;
        s_coeff2 = sqrt(square_det_J + 0.3*t2) / abs_det_J;
        opac_coeff = 1.0 / (s_coeff1 * s_coeff2);
    }
    /////////////////
    

    vec3 w_offset1 = geom_splatRS0[0]*s_coeff1*cutoff;
    vec3 w_offset2 = geom_splatRS1[0]*s_coeff2*cutoff;
    vec3 wps[4];
    vec3 c_pos3 = c_pos.xyz;
    wps[0] = c_pos3 + w_offset1 + w_offset2;
    wps[1] = c_pos3 - w_offset1 + w_offset2;
    wps[3] = c_pos3 - w_offset1 - w_offset2;
    wps[2] = c_pos3 + w_offset1 - w_offset2;
    vec2 uvs[4];
    uvs[0] = vec2(cutoff, cutoff);
    uvs[1] = vec2(-cutoff, cutoff);
    uvs[3] = vec2(-cutoff, -cutoff);
    uvs[2] = vec2(cutoff, -cutoff);

    for (int i = 0; i < 4; i++) {
        gl_Position = pvMat * vec4(wps[i], 1.0f); // keep gl_Position.w to ensure perspective correct uv interpolation
        frag_color = vec4(geom_color[0].rgb, opacity*opac_coeff);
        frag_uv = uvs[i];
        EmitVertex();
    }

    EndPrimitive();
}
