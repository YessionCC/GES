
cmake_minimum_required(VERSION 3.0)

project(ges_renderer)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++17") 
set(CMAKE_BUILD_TYPE "Release")

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*)
include_directories(${PROJECT_SOURCE_DIR}/include)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 

include_directories(${PROJECT_SOURCE_DIR}/include/glm)
add_subdirectory(${PROJECT_SOURCE_DIR}/include/glm)

include_directories(${PROJECT_SOURCE_DIR}/include/glad/include)

include_directories(${PROJECT_SOURCE_DIR}/include/imgui)
add_subdirectory(${PROJECT_SOURCE_DIR}/include/imgui)

set(EXE_FILE main)

add_executable(${EXE_FILE} 
  ${PROJECT_SOURCE_DIR}/main/${EXE_FILE}.cpp 
  ${PROJECT_SOURCE_DIR}/include/glad/src/glad.c
  ${SRC_FILES}
)

target_link_libraries(${EXE_FILE} 
  glfw imgui dl
)

set(EXE_FILE eval)

add_executable(${EXE_FILE} 
  ${PROJECT_SOURCE_DIR}/main/${EXE_FILE}.cpp 
  ${PROJECT_SOURCE_DIR}/include/glad/src/glad.c
  ${SRC_FILES}
)

target_link_libraries(${EXE_FILE} 
  glfw imgui dl
)
