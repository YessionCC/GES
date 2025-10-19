#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "camera.h"
#include "renderer.h"


class Viewer {
private:
  int windowHeight = 1080;
  int windowWidth = 1920;
  ViewportCamera vpCam;

  //Postprocess* pp;
  GLFWwindow* window;

  std::shared_ptr<SplatRenderer> renderer = nullptr;


public:
  float scrollSensitivity = 2.0f;
  float translateSensitivity = 0.1f;
  float rotateSensitivity = 0.4f;
  float travelSpeed = 0.1f;

private:
  float deltaTime = 0.0f;

private:
  // input state
  double mouseX = 0, mouseY = 0;
  bool mouseMidleButtPress = false;
  bool mouseRightButtPress = false;
  bool mouseLeftButtPress = false;

private:
  Viewer() {
    vpCam.wDivH = 1.0f*windowWidth/windowHeight;
  }
  void calcDeltaTime();

  void processInput();

public:
  Viewer(const Viewer& rhs) = delete;
  Viewer& operator=(const Viewer& rhs) = delete;

  void resizeWindow(int windowHeight, int windowWidth);

  void Init();
  void Start();
  void SetRenderer(std::shared_ptr<SplatRenderer> renderer) {this->renderer = renderer;}

  void Capture(std::string savePath);

  void Clear();

  inline int GetHeight() const {return windowHeight;}
  inline int GetWidth() const {return windowWidth;}
  inline float GetDeltaTime() const {return deltaTime;}

  static inline Viewer& GetInstance() {
    static Viewer instance;
    return instance;
  }

  static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
  static void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
  static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
  static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
  static void file_drop_in(GLFWwindow* window, int cnt, const char** paths);

};