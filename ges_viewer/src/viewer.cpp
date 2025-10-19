#include "viewer.h"
#include "editor.h"
#include "log.h"

#include "util.h"
#include "framebuffer.h"


using namespace std;

void Viewer::framebuffer_size_callback(GLFWwindow* window, int width, int height) {

  Viewer& cvp = GetInstance();
  cvp.resizeWindow(height, width);
  glViewport(0, 0, width, height);
}

void Viewer::mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
  static bool firstMouse = true;
  Viewer& cvp = GetInstance();

  if(firstMouse) {
    cvp.mouseX = xposIn;
    cvp.mouseY = yposIn;
    firstMouse = false;
  }
  float xoffset = xposIn - cvp.mouseX;
  float yoffset = cvp.mouseY - yposIn; // reversed y-coordinates

  cvp.mouseX = xposIn;
  cvp.mouseY = yposIn;

  if(cvp.mouseRightButtPress) {
    float scale = 0.03*cvp.rotateSensitivity;
    cvp.vpCam.rotateAroundFocus(xoffset*scale, yoffset*scale);
  }
  if(cvp.mouseMidleButtPress) {
    float scale = 0.03*cvp.translateSensitivity;
    cvp.vpCam.translateInXYPlane(-xoffset*scale, -yoffset*scale);
  }
}

void Viewer::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
  Viewer& cvp = GetInstance();
  cvp.vpCam.translateAlongZ(-yoffset*cvp.scrollSensitivity*0.03);
}

void Viewer::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
  Viewer& cvp = GetInstance();
  static double pressX, pressY, pressTime;
	if (action == GLFW_PRESS || action == GLFW_RELEASE) {
    if(action == GLFW_PRESS) {
      glfwGetCursorPos(window, &pressX, &pressY);
      pressTime = glfwGetTime();
    }
    else {
      double releaseX, releaseY, releaseTime;
      glfwGetCursorPos(window, &releaseX, &releaseY);
      releaseTime = glfwGetTime();
      if(releaseTime-pressTime < 0.2 || (pressX==releaseX && pressY==releaseY)) {
        // now cvp.mouseLeftButtPress ... are still true, we can use it to distinguish
        // which mouse click
      }
    }
    switch(button) {
    case GLFW_MOUSE_BUTTON_LEFT:
      cvp.mouseLeftButtPress = action;
      break;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      cvp.mouseMidleButtPress = action;
      break;
    case GLFW_MOUSE_BUTTON_RIGHT:
      cvp.mouseRightButtPress = action;
      break;
    default:
      return;
    }
  }
}

void Viewer::processInput() {
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    vpCam.goForwardAlongZ(deltaTime*1.5);
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    vpCam.goForwardAlongZ(-deltaTime*1.5);
  }
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    vpCam.goAlongX(-deltaTime*1.2);
  }
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    vpCam.goAlongX(deltaTime*1.2);
  }
}

void Viewer::file_drop_in(GLFWwindow* window, int cnt, const char** paths) {
  if (cnt != 1) {
    Log::W("Please drop only one file");
    return;
  }
  std::string fn(paths[0]);
  std::string post = fn.substr(fn.find_last_of('.')+1, 3);
  if (post != "ply") {
    Log::W("Please drop a .ply file");
    return;
  }
  std::shared_ptr<GaussianCloud> pc = std::make_shared<GaussianCloud>();
  pc->ImportPly(fn);
  Viewer& cvp = GetInstance();
  cvp.renderer->SetGaussianTarget(pc);
}

void Viewer::resizeWindow(int windowHeight, int windowWidth) {
  this->windowHeight = windowHeight;
  this->windowWidth = windowWidth;
  this->vpCam.wDivH = 1.0f*windowWidth/windowHeight;

  Editor::getInstance().curWinHeight = windowHeight;
  Editor::getInstance().curWinWidth = windowWidth;

  if(renderer)
    renderer->Resize(windowWidth, windowHeight);
}

void Viewer::calcDeltaTime() {
  float currentFrame = static_cast<float>(glfwGetTime());
  static float lastFrame = currentFrame;
  deltaTime = currentFrame - lastFrame;
  lastFrame = currentFrame;
}

void Viewer::Init() {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  //glfwWindowHint(GLFW_SAMPLES, antialiasing); // MSAA

  window = glfwCreateWindow(windowWidth, windowHeight, "Viewer", NULL, NULL);
  
  if (window == NULL) {
    Log::E("Failed to create GLFW window\n");
    glfwTerminate();
    return;
  }
  
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetMouseButtonCallback(window, mouse_button_callback);
  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetDropCallback(window, file_drop_in);

  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    Log::E("Failed to initialize GLAD\n");
  }
  Editor::getInstance().init(window, windowWidth, windowHeight);
}

void Viewer::Clear() {

  glViewport(0, 0, windowWidth, windowHeight);

}

void savePath(std::vector<glm::mat4>& poses) {
  Log::I("start save path\n");
  std::ofstream file("path.txt");
  for (glm::mat4& pose: poses) {
    file << pose[0][0] << " " << pose[0][1] << " " << pose[0][2] << " " << pose[0][3] << " " 
        << pose[1][0] << " " << pose[1][1] << " " << pose[1][2] << " " << pose[1][3] << " " 
        << pose[2][0] << " " << pose[2][1] << " " << pose[2][2] << " " << pose[2][3] << " " 
        << pose[3][0] << " " << pose[3][1] << " " << pose[3][2] << " " << pose[3][3] << " \n"; 
  }
  file.close();
  Log::I("path saved\n");
}

void Viewer::Capture(std::string savePath) {
  SaveAsImage(savePath, 0, windowWidth, windowHeight, 4);
}

void Viewer::Start() {

  Camera::SetCurrentCamera(&vpCam);

  Editor& editor = Editor::getInstance();
  editor.curWinHeight = windowHeight;
  editor.curWinWidth = windowWidth;

  Serializer::loadAll(GetRootPath() + "viewer_config.ini");

  glfwSetWindowSize(window, windowWidth, windowHeight);

  std::vector<glm::mat4> poses;
  bool poseSaved = true;

  while (!glfwWindowShouldClose(window)) {
    calcDeltaTime();

    glfwSwapInterval((int)editor.Vsync);

    glfwSetWindowTitle(window, 
      ("Viewer, fps: "+std::to_string(int(1.0/deltaTime))).c_str());

    processInput();

    if (editor.renderMode == 0) renderer->SetRenderAll();
    else if (editor.renderMode == 1) renderer->SetOnlyRenderGeo();
    else if (editor.renderMode == 2) renderer->SetOnlyRenderTex();
    renderer->SetSampleNum(editor.sampleNum);
    renderer->SetUseFilter(editor.use_filter);
    renderer->SetPreciseUV(editor.precise_uv);
    renderer->opac_thr = editor.opac_thr;
    editor.tex_dimension = renderer->GetTexDimension();

    if (editor.capturePath) {
      poseSaved = false;
      poses.push_back(vpCam.GetViewMatrix());
    }
    else {
      if (!poseSaved) {
        savePath(poses);
        poses.clear();
        poseSaved = true;
      }
    }

    if (renderer->canRender()) {
      renderer->Render(
      vpCam.GetViewMatrix(), 
      vpCam.GetProjectionMatrix(),
      vpCam.GetZNearFar());

      renderer->BlitToDefault();
    }

    if (editor.captureButtOn) {
      Capture("s.png");
      editor.captureButtOn = false;
    }

    editor.update();
    
    glfwSwapBuffers(window);
    glfwPollEvents();

  }
  
  Serializer::dumpAll(GetRootPath() + "viewer_config.ini");
  editor.terminate();
  //glfwDestroyWindow(window);
  //glfwTerminate();
}


