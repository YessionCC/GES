#include <iostream>
#include <memory>
#include <sstream>
#include <ostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "gaussianCloud.h"
#include "renderer.h"
#include "eval_viewer.h"

// sample_num
// W, H
// bgc
// plypath, mat_path, out_dir

int main(int argc, char *argv[]) {

  int sample_num = atoi(argv[1]); // deprecated
  int W = atoi(argv[2]);
  int H = atoi(argv[3]);
  float bgc1 = atof(argv[4]);
  float bgc2 = atof(argv[5]);
  float bgc3 = atof(argv[6]);
  std::string ply_path = std::string(argv[7]);
  std::string mat_path = std::string(argv[8]);
  std::string out_dir = std::string(argv[9]);

  EViewer& viewer = EViewer::GetInstance();
  viewer.Init(W, H);

  std::shared_ptr<GaussianCloud> pc = std::make_shared<GaussianCloud>();
  pc->ImportPly(ply_path);

  std::shared_ptr<SplatRenderer> renderer = std::make_shared<SplatRenderer>();
  renderer->Init(4, W, H);
  renderer->SetGaussianTarget(pc);
  renderer->SetUseTimer(true);

  viewer.SetRenderer(renderer);

  std::vector<glm::mat4> projs;
  std::vector<glm::mat4> views;
  glm::mat4 temp;
  std::ifstream matfile(mat_path);
  while(!matfile.eof()) {
    for(int i = 0; i<4; i++) for(int j = 0; j<4; j++) {
        matfile >> temp[i][j];
    }
    projs.push_back(temp);
    for(int i = 0; i<4; i++) for(int j = 0; j<4; j++) {
        matfile >> temp[i][j];
    }
    views.push_back(temp);
  }
  projs.pop_back();
  views.pop_back();

  bool save_im = true;
  if(out_dir == "NOT_SAVE") save_im = false;

  viewer.Start(projs, views, save_im, out_dir);

  return 0;

}