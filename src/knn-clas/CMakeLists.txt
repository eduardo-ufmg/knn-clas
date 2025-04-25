# knn-specific library
add_library(knn_clas_lib STATIC
  kNSSpred.cpp
)

target_include_directories(knn_clas_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(knn_clas_lib PRIVATE common_lib)

# Fit executable
add_executable(knn-fit knn-fit.cpp)
target_link_libraries(knn-fit PRIVATE knn_clas_lib)

# Pred executable
add_executable(knn-pred knn-pred.cpp)
target_link_libraries(knn-pred PRIVATE knn_clas_lib)