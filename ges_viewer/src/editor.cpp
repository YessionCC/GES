#include "editor.h"
#include "viewer.h"
#include "camera.h"

void Editor::init(GLFWwindow* window, int w, int h) {
  ImGui::CreateContext();     // Setup Dear ImGui context
  ImGui::StyleColorsDark();       // Setup Dear ImGui style
  ImGui_ImplGlfw_InitForOpenGL(window, true);     // Setup Platform/Renderer backends
  ImGui_ImplOpenGL3_Init("#version 450");
  
}


void Editor::update() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  editorDraw();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Editor::terminate() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

void Editor::dump(std::ofstream& file) {
  
}

void Editor::load(std::ifstream& file) {
  
}


void Editor::editorDraw() {

  ImGui::SetNextWindowBgAlpha(0.4);
  ImGui::Begin("Basic");

  ImGui::Text("%dD-GES", tex_dimension);

  ImGui::Text("Render mode");
  ImGui::RadioButton("Full", &renderMode, 0);
  ImGui::RadioButton("Only Surfel", &renderMode, 1);
  ImGui::RadioButton("Only Gaussian", &renderMode, 2);

  ImGui::Text("Resolution");
  ImGui::Text("W*H = %d * %d", curWinWidth, curWinHeight);

  ImGui::Checkbox("Vsync", &Vsync);
  if (tex_dimension == 2)
    ImGui::Checkbox("Use filter", &use_filter);

  ImGui::Checkbox("Precise UV", &precise_uv);
  ImGui::Text("Multisample");
  ImGui::RadioButton("None", &sampleNum, 1); ImGui::SameLine();
  ImGui::RadioButton("4x", &sampleNum, 4); ImGui::SameLine();
  ImGui::RadioButton("16x", &sampleNum, 16);

  ImGui::SliderFloat("OpacThr", &opac_thr, 0, 1.5);

  captureButtOn = ImGui::Button("Capture image");

  if (!capturePath) {
    capturePath = ImGui::Button("Capture path");
  }
  else {
    capturePath = !ImGui::Button("Stop capture path");
  }
  
  ImGui::End();

  ImGui::SetNextWindowBgAlpha(0.4);
  ImGui::Begin("Camera");

  Camera* cam = Camera::CurrentCam();
  float fov_deg = cam->fov;
  fov_deg = glm::radians(fov_deg);
  ImGui::SliderAngle("FOV", &fov_deg, 10.0f, 160.0f);
  cam->fov = glm::degrees(fov_deg);

  ImGui::SliderFloat("Near", &cam->nearZ, 0.0001f, 2.0f);
  ImGui::SliderFloat("Far", &cam->farZ, 10.1f, 10000.0f);

  glm::vec3 up = cam->WorldUp;
  ImGui::SliderFloat3("World Up", (float*)(&up), -1, 1);
  cam->WorldUp = glm::normalize(up);

  ImGui::End();
}