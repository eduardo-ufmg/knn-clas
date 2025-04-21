# Fit executable
add_executable(nn-fit
  fit.cpp
)

target_link_libraries(nn-fit PRIVATE
  common_lib
)

# Label executable
add_executable(nn-pred
  pred.cpp
)

target_link_libraries(nn-pred PRIVATE
  common_lib
)
