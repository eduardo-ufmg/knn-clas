# Fit executable
add_executable(knn-fit
  knn-fit.cpp
)

target_link_libraries(knn-fit PRIVATE
  common_lib
)

# Pred executable
add_executable(knn-pred
  knn-pred.cpp
)

target_link_libraries(knn-pred PRIVATE
  common_lib
)
