project(hwcv_kernels_cvt_color)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_vitis REQUIRED)
find_package(vitis_common REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core)

set(XF_WIDTH 1024)
set(XF_HEIGHT 768)
set(XF_USE_URAM 1)

if(DEFINED ROS_ACCELERATION AND DEFINED ROS_VITIS)
  set(CVT_COLOR_DEMOSAICING_NAME "demosaicing_accel")
  set(CVT_COLOR_BGR2GRAY_NAME "bgr2gray_accel")
  set(CVT_COLOR_BINARY_FILE
      "/lib/firmware/xilinx/${PROJECT_NAME}/${PROJECT_NAME}.xclbin")

  configure_file(cmake/config.in generated/cvt_color_config.hpp @ONLY)

  if(NOT NOKERNEL AND NOT NO_CVT_COLOR_KERNEL)

    set(SOURCE_DIR src/kernels/cvt_color/)
    message(WARNING "Compiling for FPGA. This might take a while")
    vitis_acceleration_kernel(
      NAME
      ${CVT_COLOR_DEMOSAICING_NAME}
      FILE
      ${SOURCE_DIR}/src/demosaicing_accel.cpp
      CONFIG
      ${SOURCE_DIR}/cfg/kv260.cfg
      INCLUDE
      ${CMAKE_INSTALL_PREFIX}/include
      ${CMAKE_CURRENT_BINARY_DIR}/generated
      TYPE
      hw)
    vitis_acceleration_kernel(
      NAME
      ${CVT_COLOR_BGR2GRAY_NAME}
      FILE
      ${SOURCE_DIR}/src/bgr2gray_accel.cpp
      CONFIG
      ${SOURCE_DIR}/cfg/kv260.cfg
      INCLUDE
      ${CMAKE_INSTALL_PREFIX}/include
      ${CMAKE_CURRENT_BINARY_DIR}/generated
      TYPE
      hw)

    vitis_link_kernel(
      OUTPUT
      ${PROJECT_NAME}
      KERNELS
      ${CVT_COLOR_DEMOSAICING_NAME}
      ${CVT_COLOR_BGR2GRAY_NAME}
      CONFIG
      ${SOURCE_DIR}/cfg/kv260.cfg)
  endif()

else()
  configure_file(cmake/config.in generated/cvt_color_config.hpp @ONLY)

endif()

add_library(${PROJECT_NAME} STATIC src/demosaicing_accel.cpp
                                   src/bgr2gray_accel.cpp src/accel.cpp)
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include> src/
         ${CMAKE_CURRENT_BINARY_DIR}/generated)

target_link_libraries(${PROJECT_NAME} PUBLIC vitis_common::vitis_common_header)

install(DIRECTORY include/ DESTINATION include)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES
  DESTINATION include)
