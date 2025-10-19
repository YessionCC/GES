cmake_minimum_required(VERSION 3.0)
project(imgui)

file(GLOB imgui_sources CONFIGURE_DEPENDS  "./*.cpp")
file(GLOB imgui_impl CONFIGURE_DEPENDS  
"./backends/imgui_impl_glfw.cpp" 
"./backends/imgui_impl_glfw.h"
"./backends/imgui_impl_opengl3.cpp" 
"./backends/imgui_impl_opengl3.h")
add_library(imgui STATIC ${imgui_sources} ${imgui_impl})
target_include_directories(imgui PUBLIC ".")
target_link_libraries(imgui PUBLIC glfw)
