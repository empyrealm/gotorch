cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gotorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_definitions(-DGOTORCH_EXPORT)

# Enable CUDA support if available
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA found - enabling GPU features")
else()
    message(STATUS "CUDA not found - GPU features disabled")
endif()

include_directories(../internal/torch)

add_library(gotorch SHARED
	../internal/torch/api.h ../internal/torch/exception.hpp
	../internal/torch/loss.h loss.cpp
	../internal/torch/operator.h operator.cpp
	../internal/torch/optimizer.h optimizer.cpp
	../internal/torch/module.h module.cpp
	../internal/torch/tensor.h tensor.cpp
	conv.cpp
	utils.cpp)
target_link_libraries(gotorch "${TORCH_LIBRARIES}")
set_property(TARGET gotorch PROPERTY CXX_STANDARD 17)