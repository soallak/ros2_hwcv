project(hwcv_kernels_cvt_color)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_vitis REQUIRED)
find_package(vitis_common REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core)

set(XF_WIDTH 1024)
set(XF_HEIGHT 768)
set(XF_USE_URAM 1)

if(DEFINED ROS_ACCELERATION AND DEFINED ROS_VITIS)

  set(CVT_COLOR_DEMOSAICING_NAME "demosaicing_accel")
  set(CVT_COLOR_BGR2GRAY_NAME "bgr2gray_accel")
  set(CVT_COLOR_BINARY_FILE ${HWCV_KERNELS_BINARY_FILE})

  configure_file(cmake/config.in generated/cvt_color_config.hpp @ONLY)

  if(NOT NOKERNEL AND NOT NO_CVT_COLOR_KERNEL)
    message(WARNING "Compiling for FPGA. This might take a while")

    set(SOURCE_DIR src/kernels/cvt_color/)
    set(HWCV_KERNERLS_CVT_COLOR
        ${CVT_COLOR_DEMOSAICING_NAME} ${CVT_COLOR_BGR2GRAY_NAME}
        PARENT_SCOPE)

    vitis_acceleration_kernel(
      NAME
      ${CVT_COLOR_DEMOSAICING_NAME}
      FILE
      ${SOURCE_DIR}/src/demosaicing_accel.cpp
      CONFIG
      ${HWCV_KERNELS_CFG}
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
      ${HWCV_KERNELS_CFG}
      INCLUDE
      ${CMAKE_INSTALL_PREFIX}/include
      ${CMAKE_CURRENT_BINARY_DIR}/generated
      TYPE
      hw)
  endif()
else()
  configure_file(cmake/config.in generated/cvt_color_config.hpp @ONLY)
endif()

add_library(${PROJECT_NAME} SHARED src/demosaicing_accel.cpp
                                   src/bgr2gray_accel.cpp src/accel.cpp)
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>)

target_link_libraries(${PROJECT_NAME} PUBLIC opencv_core
                                             vitis_common::vitis_common_header)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated/ DESTINATION include)
