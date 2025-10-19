#include "framebuffer.h"

#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "log.h"

void ClearFrameBuffer(GLuint fbo, glm::vec4 clr_col) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClearColor(clr_col.r, clr_col.g, clr_col.b, clr_col.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void BlitFrameBuffer(GLuint fbo_src, GLuint fbo_dst, int w, int h, bool blit_depth) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_src);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_dst);	
    if (blit_depth)			
        glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, 
            GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    else 
        glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, 
            GL_COLOR_BUFFER_BIT, GL_NEAREST);
} 

void BindFBO(GLuint fbo) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);  
}

GLuint CreateMS_F16(int w, int h, int sample_num, GLuint& colorBuffer) {
    GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);  
    // Create a multisampled color attachment texture
    glGenTextures(1, &colorBuffer);

    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, colorBuffer);
    // Do not use GL_RGB, it will clamp value between 0~1, but we have negative RGB value
    // if use RGB32F, will decrease the performance
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, sample_num, GL_RGBA16F, w, h, GL_TRUE);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, colorBuffer, 0);
    // Create a renderbuffer object for depth attachment
    GLuint rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo); 
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, sample_num, GL_DEPTH_COMPONENT, w, h); 
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo); 

    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    GL_ERROR_CHECK("FrameBuffer::CreateMSFBO end");

    return framebuffer;
}

GLuint CreateF16_C2(int w, int h, GLuint& colorBuffer1, GLuint& colorBuffer2) {
    GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);  

    glGenTextures(1, &colorBuffer1);
    glBindTexture(GL_TEXTURE_2D, colorBuffer1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorBuffer1, 0);

    glGenTextures(1, &colorBuffer2);
    glBindTexture(GL_TEXTURE_2D, colorBuffer2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, colorBuffer2, 0);

    glBindTexture(GL_TEXTURE_2D, 0);

    // Create a renderbuffer object for depth attachment
    GLuint rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo); 
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h); 
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo); 

    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    GL_ERROR_CHECK("FrameBuffer::CreateFloatFBO end");
    return framebuffer;
}

GLuint CreateUInt8(int w, int h) {
    GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);  
    // Create a multisampled color attachment texture
    GLuint textureColorBuffer;
    glGenTextures(1, &textureColorBuffer);

    glBindTexture(GL_TEXTURE_2D, textureColorBuffer);
    // Do not use GL_RGB, it will clamp value between 0~1, but we have negative RGB value
    // if use RGB32F, will decrease the performance
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorBuffer, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    GL_ERROR_CHECK("FrameBuffer::CreateUInt8 end");
    return framebuffer;
}

void DeleteFBO(GLuint fbo) {
    glDeleteFramebuffers(1, &fbo);
}

void SaveAsImage(std::string savePath, GLuint fbo, int w, int h, int channel, bool vflip) {
    char* data = (char*)malloc((h+1)*w*channel);
    char* data_c3 = (char*)malloc((h+1)*w*3);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    if(channel == 4)
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, data);
    else
        glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, data);
    GL_ERROR_CHECK("FrameBuffer::SaveAsImage end");

    int tot_pixels = w*h;
    for (int i = 0; i<tot_pixels; i++) {
        const int rgba_idx = i*4;
        const int rgb_idx = i*3;
        data_c3[rgb_idx] = data[rgba_idx];
        data_c3[rgb_idx+1] = data[rgba_idx+1];
        data_c3[rgb_idx+2] = data[rgba_idx+2];
    }

    if (vflip)
        stbi_flip_vertically_on_write(1);
    stbi_write_png(savePath.c_str(), w, h, 3, data_c3, 0);
    free(data);
    free(data_c3);

    Log::I("Rendering result saved\n");
}

unsigned int quadVAO = 0;
unsigned int quadVBO;
void renderQuad()
{
    if (quadVAO == 0)
    {
        float quadVertices[] = {
            // positions        // texture Coords
            -1.0f,  1.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f,
             1.0f,  1.0f, 1.0f, 1.0f,
             1.0f, -1.0f, 1.0f, 0.0f,
        };
        // setup plane VAO
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    }
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}
