# Train executable
add_executable(train
  train.cpp
)

target_link_libraries(train PRIVATE
  common_lib
)

# Label executable
add_executable(label
  label.cpp
)

target_link_libraries(label PRIVATE
  common_lib
)
