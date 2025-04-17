cmake_minimum_required(VERSION 3.10)
project(nn-clas)

set(CMAKE_CXX_STANDARD 20)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Find Protobuf
find_package(Protobuf REQUIRED)

# Add subdirectories
add_subdirectory(src/proto)
add_subdirectory(src/common)
add_subdirectory(src/app)
