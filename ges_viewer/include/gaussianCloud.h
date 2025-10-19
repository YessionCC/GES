/*
    Copyright (c) 2024 Anthony J. Thibault
    This software is licensed under the MIT License. See LICENSE for more details.
*/

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "binaryattribute.h"

class GaussianCloud
{
public:

    GaussianCloud();

    bool ImportPly(const std::string& plyFilename);

    size_t GetNumGaussians() const { return numGaussians; }
    size_t GetNumTexGaussians() const { return numTexGaussians; }
    size_t GetNumGeoGaussians() const { return numGeoGaussians; }
    size_t GetStride() const { return gaussianSize; }
    size_t GetTotalSize() const { return GetNumGaussians() * gaussianSize; }
    size_t GetTexSize() const { return GetNumTexGaussians() * gaussianSize; }
    size_t GetGeoSize() const { return GetNumGeoGaussians() * gaussianSize; }
    void* GetTexDataPtr() { return data.get(); }
    const void* GetTexDataPtr() const { return data.get(); }
    void* GetGeoDataPtr() { return reinterpret_cast<void*>(reinterpret_cast<char*>(data.get())+GetTexSize()); }
    const void* GetGeoDataPtr() const { return reinterpret_cast<const void*>(reinterpret_cast<const char*>(data.get())+GetTexSize()); }

    size_t GetTexDimension() const {return tex_dimension;}
    const BinaryAttribute& GetPosWithAlphaAttrib() const { return posWithAlphaAttrib; }
    const BinaryAttribute& GetR_SH0Attrib() const { return r_sh0Attrib; }
    const BinaryAttribute& GetR_SH1Attrib() const { return r_sh1Attrib; }
    const BinaryAttribute& GetR_SH2Attrib() const { return r_sh2Attrib; }
    const BinaryAttribute& GetR_SH3Attrib() const { return r_sh3Attrib; }
    const BinaryAttribute& GetG_SH0Attrib() const { return g_sh0Attrib; }
    const BinaryAttribute& GetG_SH1Attrib() const { return g_sh1Attrib; }
    const BinaryAttribute& GetG_SH2Attrib() const { return g_sh2Attrib; }
    const BinaryAttribute& GetG_SH3Attrib() const { return g_sh3Attrib; }
    const BinaryAttribute& GetB_SH0Attrib() const { return b_sh0Attrib; }
    const BinaryAttribute& GetB_SH1Attrib() const { return b_sh1Attrib; }
    const BinaryAttribute& GetB_SH2Attrib() const { return b_sh2Attrib; }
    const BinaryAttribute& GetB_SH3Attrib() const { return b_sh3Attrib; }
    const BinaryAttribute& GetSplat_RS0Attrib() const { return splatRS_0Attrib; }
    const BinaryAttribute& GetSplat_RS1Attrib() const { return splatRS_1Attrib; }
    const BinaryAttribute& GetSplat_RS2Attrib() const { return splatRS_2Attrib; }


protected:
    void InitAttribs();

    std::shared_ptr<void> data;

    BinaryAttribute posWithAlphaAttrib;
    BinaryAttribute r_sh0Attrib;
    BinaryAttribute r_sh1Attrib;
    BinaryAttribute r_sh2Attrib;
    BinaryAttribute r_sh3Attrib;
    BinaryAttribute g_sh0Attrib;
    BinaryAttribute g_sh1Attrib;
    BinaryAttribute g_sh2Attrib;
    BinaryAttribute g_sh3Attrib;
    BinaryAttribute b_sh0Attrib;
    BinaryAttribute b_sh1Attrib;
    BinaryAttribute b_sh2Attrib;
    BinaryAttribute b_sh3Attrib;
    BinaryAttribute splatRS_0Attrib;
    BinaryAttribute splatRS_1Attrib;
    BinaryAttribute splatRS_2Attrib;

    size_t tex_dimension;
    size_t numGaussians;
    size_t gaussianSize;
    size_t numTexGaussians;
    size_t numGeoGaussians;

};
