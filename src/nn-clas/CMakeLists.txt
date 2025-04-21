# Fit executable
add_executable(knn-fit
  fit.cpp
)

target_link_libraries(knn-fit PRIVATE
  common_lib
)

# Pred executable
add_executable(knn-pred
  pred.cpp
)

target_link_libraries(knn-pred PRIVATE
  common_lib
)
