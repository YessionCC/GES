/*
    Copyright (c) 2024 Anthony J. Thibault
    This software is licensed under the MIT License. See LICENSE for more details.
*/

#include "renderer.h"

#include <glad/glad.h>

#include <glm/gtc/matrix_transform.hpp>

#include "log.h"
#include "util.h"
#include "viewer.h"
#include "framebuffer.h"


static void SetupAttrib(int loc, const BinaryAttribute& attrib, int32_t count, size_t stride)
{
    assert(attrib.type == BinaryAttribute::Type::Float);
    glVertexAttribPointer(loc, count, GL_FLOAT, GL_FALSE, (uint32_t)stride, (void*)attrib.offset);
    glEnableVertexAttribArray(loc);
}

SplatRenderer::SplatRenderer(): onlyRenderGeo(false), onlyRenderTex(false), useFilter(true), recordTime(false), use_precise_uv(false), texGaussNum(0), geoGaussNum(0), bg_color(0.0)
{
}

SplatRenderer::~SplatRenderer()
{
}

void SplatRenderer::SetGaussianTarget(std::shared_ptr<GaussianCloud> gaussianCloud) {
    texGaussNum = gaussianCloud->GetNumTexGaussians();
    geoGaussNum = gaussianCloud->GetNumGeoGaussians();

    BuildVertexArrayObject(gaussianCloud);
}

bool SplatRenderer::Init(int sample_num, int W, int H)
{
    GL_ERROR_CHECK("SplatRenderer::Init() begin");

    geoSplatProg = std::make_shared<Program>();
    geoSplat_precise_Prog = std::make_shared<Program>();
    geoResolveProg = std::make_shared<Program>();
    blurProg = std::make_shared<Program>();
    texSplatProg = std::make_shared<Program>();
    tex3DSplatProg = std::make_shared<Program>();
    normProg = std::make_shared<Program>();
    
    this->sample_num = sample_num;
    this->renderW = W; this->renderH = H;

    if (!geoSplatProg->LoadVertGeomFrag("shader/splat_vert.glsl", "shader/splat_geom_geo.glsl", "shader/splat_frag_geo.glsl"))
    {
        Log::E("Error loading geoSplatProg shaders!\n");
        return false;
    }

    if (!geoSplat_precise_Prog->LoadVertGeomFrag("shader/splat_vert.glsl", "shader/splat_geom_geo.glsl", "shader/splat_frag_geo_precise.glsl"))
    {
        Log::E("Error loading geoSplat_precise_Prog shaders!\n");
        return false;
    }

    if (!geoResolveProg->LoadVertFrag("shader/clip_vert.glsl", "shader/ms_frag.glsl"))
    {
        Log::E("Error loading geoResolveProg shaders!\n");
        return false;
    }

    if (!blurProg->LoadVertFrag("shader/clip_vert.glsl", "shader/blur_frag.glsl"))
    {
        Log::E("Error loading blurProg shaders!\n");
        return false;
    }
    
    if (!texSplatProg->LoadVertGeomFrag("shader/splat_vert.glsl", "shader/splat_geom_tex.glsl", "shader/splat_frag_tex.glsl"))
    {
        Log::E("Error loading texSplatProg shaders!\n");
        return false;
    }

    if (!tex3DSplatProg->LoadVertGeomFrag("shader/splat_vert3D.glsl", "shader/splat_geom_tex3D.glsl", "shader/splat_frag_tex3D.glsl"))
    {
        Log::E("Error loading tex3DSplatProg shaders!\n");
        return false;
    }


    if (!normProg->LoadVertFrag("shader/clip_vert.glsl", "shader/norm_frag.glsl"))
    {
        Log::E("Error loading normProg shaders!\n");
        return false;
    }
    
    ms_f16_fbo = CreateMS_F16(W, H, sample_num, ms_f16_colorBuffer);
    f16_fbo = CreateF16_C2(W, H, f16_colorBuffer1, f16_colorBuffer2);
    uint8_fbo = CreateUInt8(W, H);

    glGenQueries(1, &timeQuery);

    GL_ERROR_CHECK("SplatRenderer::Init() end");

    return true;
}

void SplatRenderer::Resize(int targetW, int targetH) {
    this->renderW = targetW; this->renderH = targetH;
    DeleteFBO(ms_f16_fbo); DeleteFBO(f16_fbo); DeleteFBO(uint8_fbo);
    ms_f16_fbo = CreateMS_F16(targetW, targetH, sample_num, ms_f16_colorBuffer);
    f16_fbo = CreateF16_C2(targetW, targetH, f16_colorBuffer1, f16_colorBuffer2);
    uint8_fbo = CreateUInt8(targetW, targetH);
}


void SplatRenderer::Render(const glm::mat4& viewMat, const glm::mat4& projMat, const glm::vec2& zNearFar)
{
    GL_ERROR_CHECK("SplatRenderer::Render() begin");

    glm::mat4 cameraMat = glm::inverse(viewMat);
    glm::vec3 eye = glm::vec3(cameraMat[3]);
    bool contain_depth = false;

    if (recordTime) {
        glBeginQuery(GL_TIME_ELAPSED, timeQuery);
    }

    if (geoGaussNum > 0 && !onlyRenderTex) {
        BindFBO(ms_f16_fbo);
        // the bg for linear depth is 0 to determine if the pixel is bg
        ClearFrameBuffer(ms_f16_fbo, glm::vec4(bg_color.r, bg_color.g, bg_color.b, 0.0f));

        // depth test and write on, blend off
        glEnable(GL_DEPTH_TEST); // 
        glDisable(GL_BLEND); 

        if (use_precise_uv) {
            geoSplat_precise_Prog->Bind();
            geoSplat_precise_Prog->SetUniform("viewMat", viewMat);
            geoSplat_precise_Prog->SetUniform("projMat", projMat);
            geoSplat_precise_Prog->SetUniform("eye", eye);
            geoSplat_precise_Prog->SetUniform("sampleNum", sample_num);
            geoSplat_precise_Prog->SetUniform("invViewMat", glm::inverse(viewMat));
            geoSplat_precise_Prog->SetUniform("invProjMat", glm::inverse(projMat));
            geoSplat_precise_Prog->SetUniform("viewport_size", glm::vec2(renderW, renderH));
        }
        else {
            geoSplatProg->Bind();
            geoSplatProg->SetUniform("viewMat", viewMat);
            geoSplatProg->SetUniform("projMat", projMat);
            geoSplatProg->SetUniform("eye", eye);
            geoSplatProg->SetUniform("sampleNum", sample_num);
        }

        geoSplatVao->Bind();
        glDrawElements(GL_POINTS, geoGaussNum, GL_UNSIGNED_INT, nullptr);
        geoSplatVao->Unbind();

        BindFBO(f16_fbo);
        ClearFrameBuffer(f16_fbo, bg_color);
        geoResolveProg->Bind();
        geoResolveProg->SetUniform("sampleNum", sample_num);
        geoResolveProg->SetUniform("zNear", zNearFar.x);
        geoResolveProg->SetUniform("zFar", zNearFar.y);
        geoResolveProg->SetTex2DMS("msColorDepth", ms_f16_colorBuffer, 0);
        renderQuad();

        glDisable(GL_DEPTH_TEST);
        blurProg->Bind();
        blurProg->SetTex2D("colorMap", f16_colorBuffer1, 0);
        blurProg->SetUniform("texSize", glm::ivec2(renderW, renderH));
        if (onlyRenderGeo) {
            GLenum drawBuffers[1] = {GL_COLOR_ATTACHMENT0};
            glDrawBuffers(1, drawBuffers);
        }
        else {
            GLenum drawBuffers[1] = {GL_COLOR_ATTACHMENT1};
            glDrawBuffers(1, drawBuffers);
        }
        renderQuad();

        contain_depth = true;

    }

    if (texGaussNum > 0 && !onlyRenderGeo) {
        BindFBO(f16_fbo);
        if (!contain_depth) {
            ClearFrameBuffer(f16_fbo, bg_color);
        }
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE); // disable depth write
        glEnable(GL_BLEND); // allow blend
        glBlendFunc(GL_ONE, GL_ONE); // add mode
        if (tex_dimension == 2) {
            texSplatProg->Bind();
            texSplatProg->SetUniform("viewMat", viewMat);
            texSplatProg->SetUniform("projMat", projMat);
            texSplatProg->SetUniform("eye", eye);
            texSplatProg->SetUniform("opac_thr", opac_thr);
            texSplatProg->SetUniform("use_filter", useFilter);
            texSplatProg->SetUniform("fxfy", glm::vec2(
                0.5f*this->renderW*projMat[0][0],
                0.5f*this->renderH*projMat[1][1]
            ));
        }
        else {
            tex3DSplatProg->Bind();
            glm::vec4 viewport(0.0,0.0,this->renderW, this->renderH);
            glm::vec4 nearFarVec(0.0f, zNearFar.x, zNearFar.y, 0.0f);
            tex3DSplatProg->SetUniform("viewMat", viewMat);
            tex3DSplatProg->SetUniform("projMat", projMat);
            tex3DSplatProg->SetUniform("viewport", viewport);
            tex3DSplatProg->SetUniform("projParams", nearFarVec);
            tex3DSplatProg->SetUniform("eye", eye);
        }

        texSplatVao->Bind();
        glDrawElements(GL_POINTS, texGaussNum, GL_UNSIGNED_INT, nullptr);
        texSplatVao->Unbind();

        // depth write off, blend off, depth test off
        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        normProg->Bind();
        if (onlyRenderTex) {
            normProg->SetTex2D("colorWeight", f16_colorBuffer1, 0);
        }
        else {
            normProg->SetTex2D("colorWeight", f16_colorBuffer2, 0);
        }
        GLenum drawBuffers[1] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, drawBuffers);
        renderQuad();
        glDepthMask(GL_TRUE); // 
    }
    
    if (recordTime) {
        glEndQuery(GL_TIME_ELAPSED);
        glGetQueryObjectuiv(timeQuery, GL_QUERY_RESULT, &renderTime);
    }

    GL_ERROR_CHECK("SplatRenderer::Render() draw");
}

void SplatRenderer::BlitToDefault() {
    BlitFrameBuffer(f16_fbo, 0, renderW, renderH, false);
}

void SplatRenderer::SaveResult(std::string path) {
    BlitFrameBuffer(f16_fbo, uint8_fbo, renderW, renderH, false);
    SaveAsImage(path, uint8_fbo, renderW, renderH);
}

void SplatRenderer::SetSampleNum(int sample_num) {
    if (sample_num == this->sample_num) return;
    this->sample_num = sample_num;
    DeleteFBO(ms_f16_fbo);
    ms_f16_fbo = CreateMS_F16(renderW, renderH, sample_num, ms_f16_colorBuffer);
}


void SplatRenderer::BuildVertexArrayObject(std::shared_ptr<GaussianCloud> gaussianCloud)
{
    const size_t stride = gaussianCloud->GetStride();
    if (texGaussNum > 0) {
        texSplatVao = std::make_shared<VertexArrayObject>();

        // allocate large buffer to hold interleaved vertex data
        texDataBuffer = std::make_shared<BufferObject>(GL_ARRAY_BUFFER,
                                                            gaussianCloud->GetTexDataPtr(),
                                                            gaussianCloud->GetTexSize(), 0);

        // build element array
        texIndexVec.clear();
        texIndexVec.reserve(texGaussNum);
        assert(texGaussNum <= std::numeric_limits<uint32_t>::max());
        for (uint32_t i = 0; i < (uint32_t)texGaussNum; i++)
        {
            texIndexVec.push_back(i);
        }
        auto texIndexBuffer = std::make_shared<BufferObject>(GL_ELEMENT_ARRAY_BUFFER, texIndexVec, GL_DYNAMIC_STORAGE_BIT);


        tex_dimension = gaussianCloud->GetTexDimension();
        texSplatVao->Bind();
        texDataBuffer->Bind();

        auto splatProg = tex_dimension == 2 ? texSplatProg : tex3DSplatProg;
        
        SetupAttrib(splatProg->GetAttribLoc("position"), gaussianCloud->GetPosWithAlphaAttrib(), 4, stride);
        SetupAttrib(splatProg->GetAttribLoc("r_sh0"), gaussianCloud->GetR_SH0Attrib(), 4, stride);
        SetupAttrib(splatProg->GetAttribLoc("g_sh0"), gaussianCloud->GetG_SH0Attrib(), 4, stride);
        SetupAttrib(splatProg->GetAttribLoc("b_sh0"), gaussianCloud->GetB_SH0Attrib(), 4, stride);
        SetupAttrib(splatProg->GetAttribLoc("r_sh1"), gaussianCloud->GetR_SH1Attrib(), 4, stride);
        SetupAttrib(splatProg->GetAttribLoc("r_sh2"), gaussianCloud->GetR_SH2Attrib(), 4, stride);
        SetupAttrib(splatProg->GetAttribLoc("r_sh3"), gaussianCloud->GetR_SH3Attrib(), 4, stride);
        SetupAttrib(splatProg->GetAttribLoc("g_sh1"), gaussianCloud->GetG_SH1Attrib(), 4, stride);
        SetupAttrib(splatProg->GetAttribLoc("g_sh2"), gaussianCloud->GetG_SH2Attrib(), 4, stride);
        SetupAttrib(splatProg->GetAttribLoc("g_sh3"), gaussianCloud->GetG_SH3Attrib(), 4, stride);
        SetupAttrib(splatProg->GetAttribLoc("b_sh1"), gaussianCloud->GetB_SH1Attrib(), 4, stride);
        SetupAttrib(splatProg->GetAttribLoc("b_sh2"), gaussianCloud->GetB_SH2Attrib(), 4, stride);
        SetupAttrib(splatProg->GetAttribLoc("b_sh3"), gaussianCloud->GetB_SH3Attrib(), 4, stride);
        SetupAttrib(splatProg->GetAttribLoc("splatRS_0"), gaussianCloud->GetSplat_RS0Attrib(), 3, stride);
        SetupAttrib(splatProg->GetAttribLoc("splatRS_1"), gaussianCloud->GetSplat_RS1Attrib(), 3, stride);

        if (tex_dimension == 3) {
            SetupAttrib(splatProg->GetAttribLoc("splatRS_2"), gaussianCloud->GetSplat_RS2Attrib(), 3, stride);
        }

        texSplatVao->SetElementBuffer(texIndexBuffer);
        texDataBuffer->Unbind();
        texSplatVao->Unbind();
    }
    
    if (geoGaussNum > 0) {
        geoSplatVao = std::make_shared<VertexArrayObject>();

        // allocate large buffer to hold interleaved vertex data
        geoDataBuffer = std::make_shared<BufferObject>(GL_ARRAY_BUFFER,
                                                            gaussianCloud->GetGeoDataPtr(),
                                                            gaussianCloud->GetGeoSize(), 0);

        // build element array
        geoIndexVec.clear();
        geoIndexVec.reserve(geoGaussNum);
        assert(geoGaussNum <= std::numeric_limits<uint32_t>::max());
        for (uint32_t i = 0; i < (uint32_t)geoGaussNum; i++)
        {
            geoIndexVec.push_back(i);
        }
        auto geoIndexBuffer = std::make_shared<BufferObject>(GL_ELEMENT_ARRAY_BUFFER, geoIndexVec, GL_DYNAMIC_STORAGE_BIT);


        geoSplatVao->Bind();
        geoDataBuffer->Bind();

        SetupAttrib(geoSplatProg->GetAttribLoc("position"), gaussianCloud->GetPosWithAlphaAttrib(), 4, stride);
        SetupAttrib(geoSplatProg->GetAttribLoc("r_sh0"), gaussianCloud->GetR_SH0Attrib(), 4, stride);
        SetupAttrib(geoSplatProg->GetAttribLoc("g_sh0"), gaussianCloud->GetG_SH0Attrib(), 4, stride);
        SetupAttrib(geoSplatProg->GetAttribLoc("b_sh0"), gaussianCloud->GetB_SH0Attrib(), 4, stride);
        SetupAttrib(geoSplatProg->GetAttribLoc("r_sh1"), gaussianCloud->GetR_SH1Attrib(), 4, stride);
        SetupAttrib(geoSplatProg->GetAttribLoc("r_sh2"), gaussianCloud->GetR_SH2Attrib(), 4, stride);
        SetupAttrib(geoSplatProg->GetAttribLoc("r_sh3"), gaussianCloud->GetR_SH3Attrib(), 4, stride);
        SetupAttrib(geoSplatProg->GetAttribLoc("g_sh1"), gaussianCloud->GetG_SH1Attrib(), 4, stride);
        SetupAttrib(geoSplatProg->GetAttribLoc("g_sh2"), gaussianCloud->GetG_SH2Attrib(), 4, stride);
        SetupAttrib(geoSplatProg->GetAttribLoc("g_sh3"), gaussianCloud->GetG_SH3Attrib(), 4, stride);
        SetupAttrib(geoSplatProg->GetAttribLoc("b_sh1"), gaussianCloud->GetB_SH1Attrib(), 4, stride);
        SetupAttrib(geoSplatProg->GetAttribLoc("b_sh2"), gaussianCloud->GetB_SH2Attrib(), 4, stride);
        SetupAttrib(geoSplatProg->GetAttribLoc("b_sh3"), gaussianCloud->GetB_SH3Attrib(), 4, stride);
        SetupAttrib(geoSplatProg->GetAttribLoc("splatRS_0"), gaussianCloud->GetSplat_RS0Attrib(), 3, stride);
        SetupAttrib(geoSplatProg->GetAttribLoc("splatRS_1"), gaussianCloud->GetSplat_RS1Attrib(), 3, stride);

        geoSplatVao->SetElementBuffer(geoIndexBuffer);
        geoDataBuffer->Unbind();
        geoSplatVao->Unbind();
    }
}
