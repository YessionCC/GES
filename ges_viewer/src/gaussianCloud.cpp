/*
    Copyright (c) 2024 Anthony J. Thibault
    This software is licensed under the MIT License. See LICENSE for more details.
*/

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string.h>

#include <glm/gtc/quaternion.hpp>

#include "gaussianCloud.h"
#include "ply.h"

#include "log.h"
#include "util.h"



struct BaseGaussianData
{
    BaseGaussianData() noexcept {}
    float posWithAlpha[4]; // center of the gaussian in object coordinates, with alpha in w
    float r_sh0[4]; // sh coeff for red channel (up to third-order)
    float g_sh0[4]; // sh coeff for green channel
    float b_sh0[4];  // sh coeff for blue channel
    float splatRS_0[3]; // 3x3 covariance matrix of the splat in object coordinates.
    float splatRS_1[3];
    float splatRS_2[3]; // used for tex3D
};

struct FullGaussianData : public BaseGaussianData
{
    FullGaussianData() noexcept {}
    float r_sh1[4];
    float r_sh2[4];
    float r_sh3[4];
    float g_sh1[4];
    float g_sh2[4];
    float g_sh3[4];
    float b_sh1[4];
    float b_sh2[4];
    float b_sh3[4];
};

static glm::mat2x3 ComputeSplatRS(float rot[4], float scale[2])
{
    glm::quat q(rot[0], rot[1], rot[2], rot[3]);
    glm::mat3 R(glm::normalize(q));
    glm::mat3 S(glm::vec3(scale[0], 0.0f, 0.0f),
                glm::vec3(0.0f, scale[1], 0.0f),
                glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat3 L = R * S;

	// center of Gaussians in the camera coordinate
	glm::mat2x3 splatRS = glm::mat2x3(
		L[0], L[1]
	);
    // u*L[0]+v*L[1]+pos = world pos
    return splatRS;
}

static glm::mat3 ComputeCovMatFromRotScale(float rot[4], float scale[3])
{
    glm::quat q(rot[0], rot[1], rot[2], rot[3]);
    glm::mat3 R(glm::normalize(q));
    glm::mat3 S(glm::vec3(scale[0], 0.0f, 0.0f),
                glm::vec3(0.0f, scale[1], 0.0f),
                glm::vec3(0.0f, 0.0f, scale[2]));
    return R * S * glm::transpose(S) * glm::transpose(R);
}

GaussianCloud::GaussianCloud() :
    numGaussians(0),
    gaussianSize(0)
{
    ;
}

bool GaussianCloud::ImportPly(const std::string& plyFilename)
{

    std::ifstream plyFile(plyFilename, std::ios::binary);
    if (!plyFile.is_open())
    {
        Log::E("failed to open %s\n", plyFilename.c_str());
        return false;
    }

    Ply ply;

    if (!ply.Parse(plyFile))
    {
        Log::E("Error parsing ply file \"%s\"\n", plyFilename.c_str());
        return false;
    }

    struct
    {
        BinaryAttribute x, y, z;
        BinaryAttribute f_dc[3];
        BinaryAttribute f_rest[45];
        BinaryAttribute opacity;
        BinaryAttribute scale[3]; // scale_2 used for tex3D
        BinaryAttribute rot[4];
    } props;

    // Get the type and offset for Gaussian properties, so we can save all properties into one array, no data is read here
    if (!ply.GetProperty("x", props.x) ||
        !ply.GetProperty("y", props.y) ||
        !ply.GetProperty("z", props.z))
    {
        Log::E("Error parsing ply file \"%s\", missing position property\n", plyFilename.c_str());
    }

    for (int i = 0; i < 3; i++)
    {
        if (!ply.GetProperty("f_dc_" + std::to_string(i), props.f_dc[i]))
        {
            Log::E("Error parsing ply file \"%s\", missing f_dc property\n", plyFilename.c_str());
        }
    }

    for (int i = 0; i < 45; i++)
    {
        if (!ply.GetProperty("f_rest_" + std::to_string(i), props.f_rest[i]))
        {
            // f_rest properties are optional
            Log::E("PLY file \"%s\", missing f_rest property\n", plyFilename.c_str());
            break;
        }
    }

    if (!ply.GetProperty("opacity", props.opacity))
    {
        Log::E("Error parsing ply file \"%s\", missing opacity property\n", plyFilename.c_str());
    }

    for (int i = 0; i < 2; i++)
    {
        if (!ply.GetProperty("scale_" + std::to_string(i), props.scale[i]))
        {
            Log::E("Error parsing ply file \"%s\", missing scale property\n", plyFilename.c_str());
        }
    }
    if (ply.GetProperty("scale_2", props.scale[2]))
    {
        Log::I("Scaling has 3 axis. Use tex3D shader\n");
        tex_dimension = 3;
    }
    else tex_dimension = 2;

    for (int i = 0; i < 4; i++)
    {
        if (!ply.GetProperty("rot_" + std::to_string(i), props.rot[i]))
        {
            Log::E("Error parsing ply file \"%s\", missing rot property\n", plyFilename.c_str());
        }
    }

    //after grab all properties attributes, we need to transfrom them to the form for shader
    // here we init attributes for shader
    InitAttribs();
    // init the buffer for data
    numGaussians = ply.GetVertexCount();
    gaussianSize = sizeof(FullGaussianData);
    FullGaussianData* fullPtr = new FullGaussianData[numGaussians];
    data.reset(fullPtr);

    // Start read data, read with 'props' format
    int i = 0;
    int numTex = -1;
    uint8_t* rawPtr = (uint8_t*)data.get();
    ply.ForEachVertex([this, &rawPtr, &i, &props, &numTex](const void* plyData, size_t size)
    {
        BaseGaussianData* basePtr = reinterpret_cast<BaseGaussianData*>(rawPtr);
        float pos[3] = {
            props.x.Read<float>(plyData),
            props.y.Read<float>(plyData),
            props.z.Read<float>(plyData)
        };
        basePtr->posWithAlpha[0] = pos[0];
        basePtr->posWithAlpha[1] = pos[1];
        basePtr->posWithAlpha[2] = pos[2];
        float alpha = glm::max(props.opacity.Read<float>(plyData), 0.0f);
        if (alpha > 1000) {
            if (numTex < 0) numTex = i;
            alpha -= 1000; // if alpha > 1000, its mod_depth for geom gs
        }
        basePtr->posWithAlpha[3] = alpha;

        FullGaussianData* fullPtr = reinterpret_cast<FullGaussianData*>(rawPtr);
        fullPtr->r_sh0[0] = props.f_dc[0].Read<float>(plyData);
        fullPtr->r_sh0[1] = props.f_rest[0].Read<float>(plyData);
        fullPtr->r_sh0[2] = props.f_rest[1].Read<float>(plyData);
        fullPtr->r_sh0[3] = props.f_rest[2].Read<float>(plyData);
        fullPtr->r_sh1[0] = props.f_rest[3].Read<float>(plyData);
        fullPtr->r_sh1[1] = props.f_rest[4].Read<float>(plyData);
        fullPtr->r_sh1[2] = props.f_rest[5].Read<float>(plyData);
        fullPtr->r_sh1[3] = props.f_rest[6].Read<float>(plyData);
        fullPtr->r_sh2[0] = props.f_rest[7].Read<float>(plyData);
        fullPtr->r_sh2[1] = props.f_rest[8].Read<float>(plyData);
        fullPtr->r_sh2[2] = props.f_rest[9].Read<float>(plyData);
        fullPtr->r_sh2[3] = props.f_rest[10].Read<float>(plyData);
        fullPtr->r_sh3[0] = props.f_rest[11].Read<float>(plyData);
        fullPtr->r_sh3[1] = props.f_rest[12].Read<float>(plyData);
        fullPtr->r_sh3[2] = props.f_rest[13].Read<float>(plyData);
        fullPtr->r_sh3[3] = props.f_rest[14].Read<float>(plyData);

        fullPtr->g_sh0[0] = props.f_dc[1].Read<float>(plyData);
        fullPtr->g_sh0[1] = props.f_rest[15].Read<float>(plyData);
        fullPtr->g_sh0[2] = props.f_rest[16].Read<float>(plyData);
        fullPtr->g_sh0[3] = props.f_rest[17].Read<float>(plyData);
        fullPtr->g_sh1[0] = props.f_rest[18].Read<float>(plyData);
        fullPtr->g_sh1[1] = props.f_rest[19].Read<float>(plyData);
        fullPtr->g_sh1[2] = props.f_rest[20].Read<float>(plyData);
        fullPtr->g_sh1[3] = props.f_rest[21].Read<float>(plyData);
        fullPtr->g_sh2[0] = props.f_rest[22].Read<float>(plyData);
        fullPtr->g_sh2[1] = props.f_rest[23].Read<float>(plyData);
        fullPtr->g_sh2[2] = props.f_rest[24].Read<float>(plyData);
        fullPtr->g_sh2[3] = props.f_rest[25].Read<float>(plyData);
        fullPtr->g_sh3[0] = props.f_rest[26].Read<float>(plyData);
        fullPtr->g_sh3[1] = props.f_rest[27].Read<float>(plyData);
        fullPtr->g_sh3[2] = props.f_rest[28].Read<float>(plyData);
        fullPtr->g_sh3[3] = props.f_rest[29].Read<float>(plyData);

        fullPtr->b_sh0[0] = props.f_dc[2].Read<float>(plyData);
        fullPtr->b_sh0[1] = props.f_rest[30].Read<float>(plyData);
        fullPtr->b_sh0[2] = props.f_rest[31].Read<float>(plyData);
        fullPtr->b_sh0[3] = props.f_rest[32].Read<float>(plyData);
        fullPtr->b_sh1[0] = props.f_rest[33].Read<float>(plyData);
        fullPtr->b_sh1[1] = props.f_rest[34].Read<float>(plyData);
        fullPtr->b_sh1[2] = props.f_rest[35].Read<float>(plyData);
        fullPtr->b_sh1[3] = props.f_rest[36].Read<float>(plyData);
        fullPtr->b_sh2[0] = props.f_rest[37].Read<float>(plyData);
        fullPtr->b_sh2[1] = props.f_rest[38].Read<float>(plyData);
        fullPtr->b_sh2[2] = props.f_rest[39].Read<float>(plyData);
        fullPtr->b_sh2[3] = props.f_rest[40].Read<float>(plyData);
        fullPtr->b_sh3[0] = props.f_rest[41].Read<float>(plyData);
        fullPtr->b_sh3[1] = props.f_rest[42].Read<float>(plyData);
        fullPtr->b_sh3[2] = props.f_rest[43].Read<float>(plyData);
        fullPtr->b_sh3[3] = props.f_rest[44].Read<float>(plyData);

        // NOTE: scale is stored in logarithmic scale in plyFile
        float scale[2] =
        {
            expf(props.scale[0].Read<float>(plyData)),
            expf(props.scale[1].Read<float>(plyData))
        };
        float rot[4] =
        {
            props.rot[0].Read<float>(plyData),
            props.rot[1].Read<float>(plyData),
            props.rot[2].Read<float>(plyData),
            props.rot[3].Read<float>(plyData)
        };

        // base gaussian is always 2D, tex gaussian is 3D only if tex_dimension == 3
        if (this->tex_dimension == 3 && numTex == -1) {
            float scale3D[3] = { scale[0], scale[1],
                expf(props.scale[2].Read<float>(plyData))
            };
            glm::mat3 V = ComputeCovMatFromRotScale(rot, scale3D);
            basePtr->splatRS_0[0] = V[0][0]; // here splatRS_x is cov3D
            basePtr->splatRS_0[1] = V[0][1];
            basePtr->splatRS_0[2] = V[0][2];
            basePtr->splatRS_1[0] = V[1][0];
            basePtr->splatRS_1[1] = V[1][1];
            basePtr->splatRS_1[2] = V[1][2];
            basePtr->splatRS_2[0] = V[2][0];
            basePtr->splatRS_2[1] = V[2][1];
            basePtr->splatRS_2[2] = V[2][2];
        }
        else {
            glm::mat2x3 T = ComputeSplatRS(rot, scale);
            basePtr->splatRS_0[0] = T[0][0];
            basePtr->splatRS_0[1] = T[0][1];
            basePtr->splatRS_0[2] = T[0][2];
            basePtr->splatRS_1[0] = T[1][0];
            basePtr->splatRS_1[1] = T[1][1];
            basePtr->splatRS_1[2] = T[1][2];
        }
        i++;
        rawPtr += gaussianSize;
    });

    if (numTex == 0) {
        Log::W("No tex Gaussians\n");
    }
    if (numTex < 0) {
        numTex = numGaussians;
        Log::W("No geo Gaussians\n");
    }
    numTexGaussians = numTex;
    numGeoGaussians = numGaussians - numTex;

    Log::I("Successfully read PLY file, %d texs, %d geos\n", numTexGaussians, numGeoGaussians);
    // after read all data into 'data', we can fetch properties by using Attribute from 'InitAttribs'
    return true;
}


void GaussianCloud::InitAttribs()
{
    // BaseGaussianData attribs
    posWithAlphaAttrib = {BinaryAttribute::Type::Float, offsetof(BaseGaussianData, posWithAlpha)};
    r_sh0Attrib = {BinaryAttribute::Type::Float, offsetof(BaseGaussianData, r_sh0)};
    g_sh0Attrib = {BinaryAttribute::Type::Float, offsetof(BaseGaussianData, g_sh0)};
    b_sh0Attrib = {BinaryAttribute::Type::Float, offsetof(BaseGaussianData, b_sh0)};
    splatRS_0Attrib = {BinaryAttribute::Type::Float, offsetof(BaseGaussianData, splatRS_0)};
    splatRS_1Attrib = {BinaryAttribute::Type::Float, offsetof(BaseGaussianData, splatRS_1)};
    splatRS_2Attrib = {BinaryAttribute::Type::Float, offsetof(BaseGaussianData, splatRS_2)};

    r_sh1Attrib = {BinaryAttribute::Type::Float, offsetof(FullGaussianData, r_sh1)};
    r_sh2Attrib = {BinaryAttribute::Type::Float, offsetof(FullGaussianData, r_sh2)};
    r_sh3Attrib = {BinaryAttribute::Type::Float, offsetof(FullGaussianData, r_sh3)};
    g_sh1Attrib = {BinaryAttribute::Type::Float, offsetof(FullGaussianData, g_sh1)};
    g_sh2Attrib = {BinaryAttribute::Type::Float, offsetof(FullGaussianData, g_sh2)};
    g_sh3Attrib = {BinaryAttribute::Type::Float, offsetof(FullGaussianData, g_sh3)};
    b_sh1Attrib = {BinaryAttribute::Type::Float, offsetof(FullGaussianData, b_sh1)};
    b_sh2Attrib = {BinaryAttribute::Type::Float, offsetof(FullGaussianData, b_sh2)};
    b_sh3Attrib = {BinaryAttribute::Type::Float, offsetof(FullGaussianData, b_sh3)};
}
