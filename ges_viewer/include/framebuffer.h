#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "util.h"


GLuint CreateMS_F16(int w, int h, int sample_num, GLuint& colorBuffer);
GLuint CreateF16_C2(int w, int h, GLuint& colorBuffer1, GLuint& colorBuffer2);
GLuint CreateUInt8(int w, int h);
void ClearFrameBuffer(GLuint fbo, glm::vec4 clr_col = glm::vec4(0.0f));
void BlitFrameBuffer(GLuint fbo_src, GLuint fbo_dst, int w, int h, bool blit_depth = true);
void SaveAsImage(std::string savePath, GLuint fbo, int w, int h, int channel = 3, bool vflip = true);
void DeleteFBO(GLuint fbo);
void BindFBO(GLuint fbo);
void renderQuad();