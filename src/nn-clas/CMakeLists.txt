# nn-specific library
add_library(nn_clas_lib STATIC
  nearestSSpred.cpp
)

target_include_directories(nn_clas_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(nn_clas_lib PRIVATE common_lib)

# Fit executable
add_executable(nn-fit nn-fit.cpp)
target_link_libraries(nn-fit PRIVATE nn_clas_lib)

# Pred executable
add_executable(nn-pred nn-pred.cpp)
target_link_libraries(nn-pred PRIVATE nn_clas_lib)
