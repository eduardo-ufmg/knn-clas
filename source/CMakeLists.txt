cmake_minimum_required(VERSION 3.10)

# Project name
project(nn_clas_study)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include all header files
include_directories(${CMAKE_SOURCE_DIR})

# Find Protobuf package
find_package(Protobuf REQUIRED)

# Include Protobuf headers
include_directories(${Protobuf_INCLUDE_DIRS})

# Collect all source files
file(GLOB SOURCES "*.cpp")

# Add executable
add_executable(nn_clas_study ${SOURCES})

# Link Protobuf library
target_link_libraries(nn_clas_study ${Protobuf_LIBRARIES})
