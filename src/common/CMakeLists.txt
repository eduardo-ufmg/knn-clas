# src/common/CMakeLists.txt
file(GLOB COMMON_SOURCES *.cpp)

add_library(common_lib STATIC
  ${COMMON_SOURCES}
)

target_link_libraries(common_lib PUBLIC
  proto_lib
)
