#include "eval_viewer.h"
#include "editor.h"
#include "log.h"

#include "util.h"
#include "framebuffer.h"
#include <glm/gtc/matrix_transform.hpp>

using namespace std;


void EViewer::Init(int W, int H) {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  //glfwWindowHint(GLFW_SAMPLES, antialiasing); // MSAA
  windowWidth = W; windowHeight = H;
  window = glfwCreateWindow(windowWidth, windowHeight, "EViewer", NULL, NULL);
  
  if (window == NULL) {
    Log::E("Failed to create GLFW window\n");
    glfwTerminate();
    return;
  }
  
  glfwMakeContextCurrent(window);

  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    Log::E("Failed to initialize GLAD\n");
  }
  Editor::getInstance().init(window, windowWidth, windowHeight);
}

void EViewer::Capture(std::string savePath) {
  SaveAsImage(savePath, 0, windowWidth, windowHeight, 4, false);
}

void EViewer::Start(std::vector<glm::mat4> projs, std::vector<glm::mat4> views, bool save_ims, std::string out_dir) {

  glfwSwapInterval(0);
  glfwSetWindowSize(window, windowWidth, windowHeight);
  char temp_str[100];
  for(int i = 0; i < views.size(); i++) {
    // modify the view and projection matrix, to make -z forward, and depth to -1~1
    glm::mat4 view_proj = views[i];
    glm::mat4 presp_proj = projs[i];
    view_proj[0][2] = -view_proj[0][2];
    view_proj[1][2] = -view_proj[1][2];
    view_proj[2][2] = -view_proj[2][2];
    view_proj[3][2] = -view_proj[3][2];
    float zNear = -presp_proj[3][2] / presp_proj[2][2];
    float zFar = zNear / (1 - 1/presp_proj[2][2]);
    presp_proj[2][2] = - (zFar + zNear) / (zFar - zNear);
	  presp_proj[2][3] = - 1;
	  presp_proj[3][2] = - (2 * zFar * zNear) / (zFar - zNear);

    renderer->Render(view_proj, presp_proj, glm::vec2(zNear, zFar));
    renderer->BlitToDefault();

    if(save_ims) {
      snprintf(temp_str, sizeof(temp_str), "%03d.png", i);
      Capture(out_dir+"/"+temp_str);
    }
    
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // test FPS, 10 rounds
  int ROUND = 10;
  float tot_renTime = 0;
  for(int round = 0; round < ROUND ; round ++) {
    for(int i = 0; i < views.size(); i++) {
      
      glm::mat4 view_proj = views[i];
      glm::mat4 presp_proj = projs[i];
      view_proj[0][2] = -view_proj[0][2];
      view_proj[1][2] = -view_proj[1][2];
      view_proj[2][2] = -view_proj[2][2];
      view_proj[3][2] = -view_proj[3][2];
      float zNear = -presp_proj[3][2] / presp_proj[2][2];
      float zFar = zNear / (1 - 1/presp_proj[2][2]);
      presp_proj[2][2] = - (zFar + zNear) / (zFar - zNear);
      presp_proj[2][3] = - 1;
      presp_proj[3][2] = - (2 * zFar * zNear) / (zFar - zNear);
      
      //float start_time = static_cast<float>(glfwGetTime());
      renderer->Render(view_proj, presp_proj, glm::vec2(zNear, zFar));
      renderer->BlitToDefault();

      glfwSwapBuffers(window);
      glfwPollEvents();

      GLuint renTime;
      renderer->GetRenderTime(renTime);
      tot_renTime += renTime * 1e-6;
    }
  }

  float avg_time = tot_renTime / (views.size() * ROUND);
  float avg_fps = 1000.0f / avg_time;
  Log::I("AVG FPS: %.2f\n", avg_fps);
  glfwDestroyWindow(window);
  glfwTerminate();
}


