# Fit executable
add_executable(nn-fit
  nn-fit.cpp
)

target_link_libraries(nn-fit PRIVATE
  common_lib
)

# Pred executable
add_executable(nn-pred
  nn-pred.cpp
)

target_link_libraries(nn-pred PRIVATE
  common_lib
)
