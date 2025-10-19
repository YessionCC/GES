/*
    Copyright (c) 2024 Anthony J. Thibault
    This software is licensed under the MIT License. See LICENSE for more details.
*/

#pragma once

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <memory>
#include <stdint.h>
#include <vector>

#include "shaderProgram.h"
#include "vertexbuffer.h"

#include "gaussianCloud.h"


class SplatRenderer
{
public:
    SplatRenderer();
    ~SplatRenderer();

    void SetOnlyRenderGeo() {
        onlyRenderGeo = true;
        onlyRenderTex = false;
    }
    void SetOnlyRenderTex() {
        onlyRenderGeo = false;
        onlyRenderTex = true;
    }
    void SetRenderAll() {
        onlyRenderGeo = false;
        onlyRenderTex = false;
    }

    bool Init(int sample_num, int W, int H);
    void SetGaussianTarget(std::shared_ptr<GaussianCloud> gaussianCloud);

    void Resize(int targetW, int targetH);

    bool canRender()  {
        return texGaussNum && geoGaussNum;
    }

    // viewport = (x, y, width, height)
    void Render(const glm::mat4& viewMat, const glm::mat4& projMat,  const glm::vec2& zNearFar);

    void BlitToDefault();

    // call after rendering
    void SaveResult(std::string path);

    void SetSampleNum(int sample_num);
    void SetBgColor(glm::vec4 bgc) {bg_color = bgc;}
    void SetUseFilter(bool use_filter) {useFilter = use_filter;}
    void SetPreciseUV(bool precise_uv) {use_precise_uv = precise_uv;}
    size_t GetTexDimension() const {return tex_dimension;}
    void SetUseTimer(bool useTimer) {recordTime = useTimer;}
    void GetRenderTime(GLuint& time_res) const {
        time_res = renderTime;
    }

    float opac_thr = 5;

protected:
    void BuildVertexArrayObject(std::shared_ptr<GaussianCloud> gaussianCloud);

    std::shared_ptr<Program> texSplatProg;
    std::shared_ptr<Program> tex3DSplatProg;
    std::shared_ptr<Program> geoResolveProg;
    std::shared_ptr<Program> blurProg;
    std::shared_ptr<Program> normProg;
    std::shared_ptr<Program> geoSplatProg;
    std::shared_ptr<Program> geoSplat_precise_Prog;

    std::shared_ptr<VertexArrayObject> texSplatVao;
    std::vector<uint32_t> texIndexVec;
    std::shared_ptr<BufferObject> texDataBuffer;

    std::shared_ptr<VertexArrayObject> geoSplatVao;
    std::vector<uint32_t> geoIndexVec;
    std::shared_ptr<BufferObject> geoDataBuffer;

    size_t texGaussNum;
    size_t geoGaussNum;
    size_t tex_dimension;

    glm::vec4 bg_color;

    int sample_num;
    int renderW, renderH;
    bool onlyRenderGeo;
    bool onlyRenderTex;
    bool useFilter;
    bool recordTime;
    bool use_precise_uv;

    GLuint ms_f16_fbo;
    GLuint ms_f16_colorBuffer;
    GLuint f16_colorBuffer1;
    GLuint f16_colorBuffer2;
    GLuint f16_fbo;
    GLuint uint8_fbo;

    GLuint timeQuery;
    GLuint renderTime;

};
