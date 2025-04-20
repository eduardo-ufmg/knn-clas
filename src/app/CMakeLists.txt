# Fit executable
add_executable(fit
  fit.cpp
)

target_link_libraries(fit PRIVATE
  common_lib
)

# Label executable
add_executable(pred
  pred.cpp
)

target_link_libraries(pred PRIVATE
  common_lib
)
