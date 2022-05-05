project(hwcv_kernels_stereo)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
endif()

find_package(ament_vitis REQUIRED)
find_package(vitis_common REQUIRED)

set(XF_WIDTH 1024)
set(XF_HEIGHT 768)
set(XF_STEREO_LBM_WSIZE 15)
set(XF_STEREO_LBM_NDISP 128)
set(XF_STEREO_LBM_NDISP_UNIT 32)
# set(XF_USE_URAM)

if(DEFINED ROS_ACCELERATION AND DEFINED ROS_VITIS)
  set(STEREO_LBM_NAME "stereo_lbm_accel")
  set(STEREO_BINARY_FILE ${HWCV_KERNELS_BINARY_FILE})
  configure_file(cmake/stereo_lbm_config.in generated/stereo_lbm_config.hpp
                 @ONLY)

  if(NOT NOKERNEL AND NOT NO_STEREO_KERNEL)
    set(SOURCE_DIR src/kernels/stereo/)
    message(WARNING "Compiling for FPGA. This might take a while")

    set(HWCV_KERNERLS_STEREO
        ${STEREO_LBM_NAME}
        PARENT_SCOPE)

    vitis_acceleration_kernel(
      NAME
      ${STEREO_LBM_NAME}
      FILE
      ${SOURCE_DIR}/src/stereo_lbm_accel.cpp
      CONFIG
      ${HWCV_KERNELS_CFG}
      INCLUDE
      ${CMAKE_INSTALL_PREFIX}/include
      ${CMAKE_CURRENT_BINARY_DIR}/generated
      TYPE
      hw)

  endif()
else()
  configure_file(cmake/stereo_lbm_config.in generated/stereo_lbm_config.hpp
                 @ONLY)
endif()

add_library(${PROJECT_NAME} STATIC src/stereo_lbm_accel.cpp src/accel.cpp)
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
