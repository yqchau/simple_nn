cmake_minimum_required(VERSION 3.5)
project(trt_inference VERSION 1.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


# find_package(CUDA REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_SOURCE_DIR}/include)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# # cudnn
# include_directories(/data/cuda/cuda-10.2/cudnn/v8.0.4/include)
# link_directories(/data/cuda/cuda-10.2/cudnn/v8.0.4/lib64)

# # tensorrt
# include_directories(/opt/tiger/demo/TensorRT-7.2.3.4/include)
# link_directories(/opt/tiger/demo/TensorRT-7.2.3.4/lib)

# tensorrt sample
include_directories(/usr/src/tensorrt/samples/common)
link_directories(/usr/src/tensorrt/samples/common)

catkin_package(
INCLUDE_DIRS include
CATKIN_DEPENDS roscpp std_msgs
)
include_directories(${catkin_INCLUDE_DIRS})
add_executable(simple_nn src/simple_nn.cpp)
target_link_libraries(simple_nn
nvinfer
cudart
nvonnxparser
${catkin_LIBRARIES}
)
