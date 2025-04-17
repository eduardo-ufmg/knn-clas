# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(nn-clas LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED)

# Include directories for project headers
include_directories(include/nn-clas)

# Add subdirectories
add_subdirectory(src/common)
add_subdirectory(src/proto)
add_subdirectory(src/app)
