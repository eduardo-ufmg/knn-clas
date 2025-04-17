# src/proto/CMakeLists.txt
add_library(proto_lib STATIC
  classifier.pb.cc
)

target_include_directories(proto_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(proto_lib PUBLIC
                      ${Protobuf_LIBRARIES}
)
