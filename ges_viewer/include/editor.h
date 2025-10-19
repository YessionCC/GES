#pragma once

#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#include <glm/glm.hpp>

#include "serializer.h"


class Editor: public Serializer {
public:
  int renderMode = 0;

  int curWinWidth, curWinHeight;
  int sampleNum = 4;

  int tex_dimension = 2;
  float opac_thr = 1.5;

  bool Vsync = true;
  bool use_filter = true;
  bool captureButtOn = false;
  bool capturePath = false;
  bool precise_uv = false;

public:
  static Editor& getInstance() {
    static Editor editor;
    return editor;
  }
  void init(GLFWwindow* window, int w, int h);
  void update();
  void terminate();

  virtual void dump(std::ofstream& file);
  virtual void load(std::ifstream& file);

private:
  void editorDraw();

};
