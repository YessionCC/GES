#include <iostream>
#include <memory>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "gaussianCloud.h"
#include "renderer.h"
#include "viewer.h"


int main() {
  int sample_num = 4;
  Viewer& viewer = Viewer::GetInstance();
  viewer.Init();

  
  std::shared_ptr<SplatRenderer> renderer = std::make_shared<SplatRenderer>();
  renderer->Init(sample_num, viewer.GetWidth(), viewer.GetHeight());

  viewer.SetRenderer(renderer);

  viewer.Start();

  return 0;

}