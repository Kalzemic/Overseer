cmake_minimum_required(VERSION 3.18)
project(OverNet CUDA CXX)

find_package(CUDAToolkit REQUIRED)

set(TENSORRT_DIR "/home/kalzemic/TensorRT/TensorRT-10.13.3.9")

include_directories(
    ${TENSORRT_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

link_directories(${TENSORRT_DIR}/lib)

add_executable(OverNet main.cpp Lookup.cu encode.cu Server.cpp)

set_target_properties(OverNet PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(OverNet PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

target_link_libraries(OverNet
    nvinfer
    CUDA::cudart
)