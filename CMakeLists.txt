cmake_minimum_required(VERSION 3.8)
project(hwcv)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
find_package(spdlog REQUIRED)
find_package(OpenCL REQUIRED)
find_package(vitis_common REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

add_subdirectory(src/kernels)

add_library(
  ${PROJECT_NAME} SHARED
  src/vitis_algorithm.cpp src/cvt_color.cpp src/stereo.cpp src/feature_fast.cpp
  src/logger.cpp src/cl_utils.cpp)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
  PRIVATE src/)

if(DEFINED ROS_ACCELERATION AND DEFINED ROS_VITIS)
  find_library(xrt_coreutil xrt_coreutil)
  find_library(xilinxopencl xilinxopencl)
  target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC opencv_core
    PRIVATE ${PROJECT_NAME}_kernels_cvt_color
            ${PROJECT_NAME}_kernels_stereo
            ${PROJECT_NAME}_kernels_feature
            Boost::filesystem
            spdlog::spdlog
            ${xrt_coreutil}
            ${xilinxopencl}
            vitis_common::vitis_common)
else()
  target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC opencv_core
    PRIVATE ${PROJECT_NAME}_kernels_cvt_color
            ${PROJECT_NAME}_kernels_stereo
            ${PROJECT_NAME}_kernels_feature
            Boost::filesystem
            spdlog::spdlog
            OpenCL::OpenCL
            vitis_common::vitis_common)
endif()
install(DIRECTORY include/ DESTINATION include)

install(
  TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_kernels_cvt_color
          ${PROJECT_NAME}_kernels_stereo ${PROJECT_NAME}_kernels_feature
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES
  DESTINATION include)

# TODO(soallak) tests are currently binaries that are invoked manually. This
# needs to change to be gtest unit tests
add_subdirectory(test)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights comment the
  # line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo) comment the line
  # when this package is in a git repo and when a copyright and license is added
  # to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rclcpp vitis_common Boost spdlog OpenCL OpenCV)
ament_export_targets("${PROJECT_NAME}Targets")
ament_package()
