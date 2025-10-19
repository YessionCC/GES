#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "camera.h"
#include "renderer.h"


class EViewer {
private:
  int windowHeight;
  int windowWidth;

  GLFWwindow* window;

  std::shared_ptr<SplatRenderer> renderer = nullptr;

private:
  float deltaTime = 0.0f;


private:
  EViewer() {}

public:
  EViewer(const EViewer& rhs) = delete;
  EViewer& operator=(const EViewer& rhs) = delete;

  void Init(int W, int H);
  void Start(std::vector<glm::mat4> projs, std::vector<glm::mat4> views, bool save_ims, std::string out_dir);
  void SetRenderer(std::shared_ptr<SplatRenderer> renderer) {this->renderer = renderer;}

  void Capture(std::string savePath);

  inline int GetHeight() const {return windowHeight;}
  inline int GetWidth() const {return windowWidth;}
  inline float GetDeltaTime() const {return deltaTime;}

  static inline EViewer& GetInstance() {
    static EViewer instance;
    return instance;
  }
};