

/*%%HEADER%%*/
/*%%DEFINES%%*/

in vec2 position;  
in vec2 uv;  

out vec2 frag_uv;


void main(void)
{
    frag_uv = uv;

    gl_Position = vec4(position, 0.0f, 1.0f);
}
