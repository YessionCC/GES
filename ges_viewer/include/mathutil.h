#pragma once

#include <cmath>
#include <iostream>
#include <glm/glm.hpp>
#include <ctime>

#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>

const float PI = M_PI;
const float PI2 = 2.0f*M_PI;


inline void PrintVec3(glm::vec3 vec) {
  std::cout<<"Vec3: "<<vec.x<<" "<<vec.y<<" "<<vec.z<<std::endl;
}

inline std::string Vec3Str(glm::vec3 vec) {
  return std::to_string(vec.x)+" "+std::to_string(vec.y)+" "+std::to_string(vec.z);
}

inline std::string Mat3Str(glm::mat3 mat) {
  return Vec3Str(mat[0])+" "+Vec3Str(mat[1])+" "+Vec3Str(mat[2]);
}

inline float NormalizeAndGetLength(glm::vec3& vec) {
  float len = glm::length(vec);
  vec /= len;
  return len;
}

// make sure vec is normalized
inline float GetThetaAroundY(glm::vec3 vec) {
  return glm::acos(vec.y);
}
// make sure vec is normalized, return 0~2pi
inline float GetPhiInXZPlane(glm::vec3 vec) {
  float phi = atan2(vec.z, vec.x);
  if(phi < 0.0f) phi += PI2;
  return phi;
}

// phi in XZ plane, theta based on Y axis
inline glm::vec3 GetVecFromPhiThetaInXZY(float phi, float theta) {
  glm::vec3 res;
  res.y = glm::cos(theta);
  float sintheta = glm::sin(theta);
  res.x = sintheta*glm::cos(phi);
  res.z = sintheta*glm::sin(phi);
  return res;
}

inline glm::mat3 GetRotationMatrix(glm::vec3 euler) {
  glm::mat4 res = glm::rotate(glm::mat4(1), euler.x, {1.0f, 0.0f, 0.0f});
  res = glm::rotate(res, euler.y, {0.0f, 1.0f, 0.0f});
  res = glm::rotate(res, euler.z, {0.0f, 0.0f, 1.0f});
  return res;
}

inline void GetSixFacesTrans(glm::vec3 center, glm::mat4 proj, glm::mat4 res[6]) {
  res[0] = proj * 
    glm::lookAt(center, center + glm::vec3(1.0,0.0,0.0), glm::vec3(0.0,-1.0,0.0));
  res[1] = proj * 
    glm::lookAt(center, center + glm::vec3(-1.0,0.0,0.0), glm::vec3(0.0,-1.0,0.0));
  res[2] = proj * 
    glm::lookAt(center, center + glm::vec3(0.0,1.0,0.0), glm::vec3(0.0,0.0,1.0));
  res[3] = proj * 
    glm::lookAt(center, center + glm::vec3(0.0,-1.0,0.0), glm::vec3(0.0,0.0,-1.0));
  res[4] = proj * 
    glm::lookAt(center, center + glm::vec3(0.0,0.0,1.0), glm::vec3(0.0,-1.0,0.0));
  res[5] = proj * 
    glm::lookAt(center, center + glm::vec3(0.0,0.0,-1.0), glm::vec3(0.0,-1.0,0.0));
}

// return H, W
inline void GetScreenPos(glm::vec3 pos, const glm::mat4& MVP, int H, int W, glm::ivec2& res) {
  glm::vec4 ndcp = MVP * glm::vec4(pos, 1.0f);
  ndcp = ndcp / ndcp.w;
  res.y = (int)((ndcp.x + 1.0f) / 2.0f * W);
  res.x = (int)((-ndcp.y + 1.0f) / 2.0f * H); // Note: here flip y-axis
}
// return H, W, range (0~H, 0~W)
inline void GetScreenPos_bound(glm::vec3 pos, const glm::mat4& MVP, int H, int W, glm::ivec2& res) {
  glm::vec4 ndcp = MVP * glm::vec4(pos, 1.0f);
  ndcp = ndcp / ndcp.w;
  res.y = (int)((ndcp.x + 1.0f) / 2.0f * W);
  res.x = (int)((-ndcp.y + 1.0f) / 2.0f * H); // Note: here flip y-axis
  res.y = glm::clamp(res.y, 0, W);
  res.x = glm::clamp(res.x, 0, H);
}

inline void getBB3Cornor(glm::vec3 cs[8], glm::vec3 pMin, glm::vec3 pMax) {
  cs[0] = pMin;
  cs[1] = glm::vec3(pMin.x, pMax.y, pMin.z);
  cs[2] = glm::vec3(pMax.x, pMax.y, pMin.z);
  cs[3] = glm::vec3(pMax.x, pMin.y, pMin.z);
  cs[4] = glm::vec3(pMin.x, pMin.y, pMax.z);
  cs[5] = glm::vec3(pMin.x, pMax.y, pMax.z);
  cs[6] = pMax;
  cs[7] = glm::vec3(pMax.x, pMin.y, pMax.z);
}

inline void getBB3ScreenPos(
  glm::vec3 pMin, glm::vec3 pMax, 
  glm::ivec2& resMin, glm::ivec2& resMax,
  const glm::mat4& MVP, int H, int W) {

  glm::vec3 cs[8];
  getBB3Cornor(cs, pMin, pMax);
  resMin = glm::ivec2(INT_MAX);
  resMax = glm::ivec2(INT_MIN);
  for(int i = 0; i<8; i++) {
    glm::ivec2 scp;
    GetScreenPos_bound(cs[i], MVP, H, W, scp);
    resMin = glm::min(resMin, scp);
    resMax = glm::max(resMax, scp);
  }
}


inline int worldDir2pixelIndex(glm::vec3 wdir, int resol) {
  glm::vec3 abs_dir = glm::abs(wdir);
  glm::vec2 uv;
  if(abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z) {
    wdir /= abs_dir.x;
    uv = wdir.x>0 ? glm::vec2(-wdir.z, -wdir.y) : glm::vec2(wdir.z, -wdir.y);
  }
  else if(abs_dir.y >= abs_dir.z) {
    wdir /= abs_dir.y;
    uv = wdir.y>0 ? glm::vec2(wdir.x, wdir.z) : glm::vec2(wdir.x, -wdir.z);
  }
  else {
    wdir /= abs_dir.z;
    uv = wdir.z>0 ? glm::vec2(wdir.x, -wdir.y) : glm::vec2(-wdir.x, -wdir.y);
  }

  uv = resol*1.0f*(0.5f*uv + 0.5f);
  int i = glm::clamp((int)uv.x, 0, resol-1); 
  int j = glm::clamp((int)uv.y, 0, resol-1); 
  return resol*j+i;
}

inline bool rasBB2inClip(
  const glm::ivec2& minRas, const glm::ivec2& maxRas, int w, int h) {
  return (minRas.x < w && maxRas.x >= 0) && (minRas.y < h && maxRas.y >= 0);
}

inline std::string GetModelNameFromPath(std::string path) {
  size_t dot_idx = path.find_last_of(".");
  size_t slash_idx = path.find_last_of("/");
  if(slash_idx == std::string::npos) slash_idx = -1;
  std::string model_name = path.substr(slash_idx+1, dot_idx - slash_idx - 1);
  return model_name;
}

inline glm::mat3 cross2Mat(glm::vec3 vec) {
  glm::mat3 res(0);
  res[0][1] = vec[2];
  res[0][2] = -vec[1];
  res[1][0] = -vec[2];
  res[1][2] = vec[0];
  res[2][0] = vec[1];
  res[2][1] = -vec[0];
  return res;
}

inline glm::mat3 rotvec2Mat(glm::vec3 rotvec) {
  float angle = glm::length(rotvec);
  glm::vec3 axis = rotvec / angle;
  glm::mat3 Axis = cross2Mat(axis);
  glm::mat3 rot = glm::mat3(1) + glm::sin(angle)*Axis+(1-glm::cos(angle))*(Axis*Axis);
  return rot;
}

inline std::string curTimeStr() {
  time_t ts; time(&ts);
  tm* t = localtime(&ts);
  int year = t->tm_year+1900;
  int month = t->tm_mon+1;
  int date = t->tm_mday;
  int h = t->tm_hour;
  int m = t->tm_min;
  int s = t->tm_sec;
  std::string res = std::to_string(month)+"_"+std::to_string(date)+"_"+
    std::to_string(h)+"_"+std::to_string(m)+"_"+std::to_string(s);
  return res;
}