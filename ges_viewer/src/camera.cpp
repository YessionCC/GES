#include "camera.h"
#include "mathutil.h"

#include <GLFW/glfw3.h>
#include <iostream>

using namespace std;

Camera* curCamera = nullptr;

Camera::Camera(glm::vec3 position, glm::vec3 lookAtPoint, glm::vec3 worldUp):
  WorldUp(glm::normalize(worldUp)), Position(position) {
  
  Front = glm::normalize(lookAtPoint - position);
  Right = glm::normalize(glm::cross(Front, WorldUp));
  Up = glm::cross(Right, Front);
}

void Camera::SetCurrentCamera(Camera* cam) {
  curCamera = cam;
}
Camera* Camera::CurrentCam() {
  return curCamera;
}


ViewportCamera::ViewportCamera(glm::vec3 position, glm::vec3 focusPoint, glm::vec3 worldUp):
  Camera(position, focusPoint, worldUp), focusPoint(focusPoint) {
  glm::vec3 dir_f2p = Position - focusPoint;
  distToFocus = NormalizeAndGetLength(dir_f2p);
  yaw = GetPhiInXZPlane(dir_f2p);
  pitch = GetThetaAroundY(dir_f2p);
}

void ViewportCamera::updateVectors() {
  Front = glm::normalize(focusPoint - Position);
  // for the pitch(theta) range in 0~2PI, we need reverse worldUp
  if(pitch > PI)
    Right = glm::normalize(glm::cross(Front, -WorldUp));
  else 
    Right = glm::normalize(glm::cross(Front, WorldUp));
  Up = glm::cross(Right, Front);
}

void ViewportCamera::translateInXYPlane(float dx, float dy) {
  glm::vec3 dir = distToFocus*(dx*Right+dy*Up);
  Position += dir;
  focusPoint += dir;
}

void ViewportCamera::rotateAroundFocus(float dYaw, float dPitch) {
  pitch += dPitch;
  if(pitch < 0.0f) pitch += PI2;
  if(pitch > PI2) pitch -= PI2;

  if(pitch > PI) yaw -= dYaw;
  else yaw += dYaw;
  if(yaw < 0.0f) yaw += PI2;
  if(yaw > PI2) yaw -= PI2;
  glm::vec3 dir = GetVecFromPhiThetaInXZY(yaw, pitch);
  Position = focusPoint + distToFocus*dir;
  updateVectors();
}

void ViewportCamera::translateAlongZ(float dz) {
  distToFocus *= (1.0f+dz);
  Position = focusPoint - distToFocus*Front;
}

void ViewportCamera::goForwardAlongZ(float dz) {
  Position += dz*Front;
  focusPoint += dz*Front;
}

void ViewportCamera::goAlongX(float dx) {
  Position += dx*Right;
  focusPoint += dx*Right;
}

void ViewportCamera::printPositionAndLookDir() const {
  glm::vec3 dir = glm::normalize(focusPoint - Position);
  cout<<"Position: "<<Position.x <<" "<<Position.y<<" "<<Position.z<<
  ", Look dir: "<<dir.x<<" "<<dir.y<<" "<<dir.z<<endl;
}

void ViewportCamera::dump(std::ofstream& file) {
  file << "ViewportCamera" << std::endl;
  file << "position " << Vec3Str(Position) << std::endl;
  file << "focuspoint " << Vec3Str(focusPoint) << std::endl;
  file << "Yaw_Pitch " << yaw << " " << pitch << std::endl;
}

void ViewportCamera::load(std::ifstream& file) {
  std::string cc;
  file >> cc;
  file >> cc >> Position.x >> Position.y >> Position.z;
  file >> cc >> focusPoint.x >> focusPoint.y >> focusPoint.z;
  file >> cc >> yaw >> pitch;
  Front = glm::normalize(focusPoint - Position);
  if(pitch > PI)
    Right = glm::normalize(glm::cross(Front, -WorldUp));
  else 
    Right = glm::normalize(glm::cross(Front, WorldUp));
  Up = glm::cross(Right, Front);
  glm::vec3 dir_f2p = Position - focusPoint;
  distToFocus = NormalizeAndGetLength(dir_f2p);
}

// pos: position, fp: focuspoint, yp: yaw and pitch
void ViewportCamera::restorePose(glm::vec3 pos, glm::vec3 fp, glm::vec2 yp) {
  this->Position = pos;
  this->focusPoint = fp;
  this->yaw = yp[0]; this->pitch = yp[1];
  Front = glm::normalize(fp - pos);
  if(pitch > PI)
    Right = glm::normalize(glm::cross(Front, -WorldUp));
  else 
    Right = glm::normalize(glm::cross(Front, WorldUp));
  Up = glm::cross(Right, Front);
  glm::vec3 dir_f2p = pos - fp;
  distToFocus = NormalizeAndGetLength(dir_f2p);
}