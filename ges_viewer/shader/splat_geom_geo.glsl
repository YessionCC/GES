
/*%%HEADER%%*/

uniform mat4 viewMat;  // used to project position into view coordinates.
uniform mat4 projMat;  // used to project view coordinates into clip coordinates.


layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec4 geom_color[];  // radiance of splat
in vec3 geom_splatRS0[]; 
in vec3 geom_splatRS1[]; 

layout(location = 0) out vec4 frag_color_depth; 
layout(location = 1) out vec2 frag_uv; 

layout(location = 2) out vec3 frag_center_world;
layout(location = 3) out vec3 frag_tangent_u;
layout(location = 4) out vec3 frag_tangent_v;


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
    
    float cutoff = 3.4f; // 3.3
    /*
    float opacity = geom_color[0].w;
    cutoff = sqrt(max(11.1f + 2.f * log(opacity), 0.000001));
    cutoff = min(3.4f, cutoff);
    */

    vec3 w_offset1 = geom_splatRS0[0]*cutoff;
    vec3 w_offset2 = geom_splatRS1[0]*cutoff;
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

    float modDepth = geom_color[0].w;

    for (int i = 0; i < 4; i++) {
        vec4 view_pos = viewMat * vec4(wps[i], 1.0f);
        // NOTE: view_pos.z is negative
        float modZ = view_pos.z - modDepth;

        gl_Position = pvMat * vec4(wps[i], 1.0f); // keep gl_Position.w to ensure perspective correct uv interpolation
        frag_color_depth = vec4(geom_color[0].rgb, modZ);
        frag_uv = uvs[i];

        frag_center_world = c_pos3;
        frag_tangent_u = geom_splatRS0[0];
        frag_tangent_v = geom_splatRS1[0];

        EmitVertex();
    }

    EndPrimitive();
}
