# knn-specific library
add_library(knn_like_lib STATIC
  ../kNSSvoting.cpp
  kNSSlikelihood.cpp
)

target_include_directories(knn_like_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(knn_like_lib PRIVATE common_lib)

# Likelihoods executable
add_executable(knn-like knn-like.cpp)
target_link_libraries(knn-like PRIVATE knn_like_lib)
