# src/app/CMakeLists.txt
# Executable for label
add_executable(label
  label.cpp
)

target_link_libraries(label PRIVATE
  common_lib
)

# Executable for train
add_executable(train
  train.cpp
)

target_link_libraries(train PRIVATE
  common_lib
)
