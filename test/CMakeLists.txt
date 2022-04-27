project(hwcv_test)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(spdlog REQUIRED)

add_executable(test_cvt_color test_cvt_color.cpp)
target_link_libraries(
  test_cvt_color PRIVATE hwcv opencv_core opencv_highgui opencv_imgproc
                         Boost::filesystem spdlog::spdlog)

add_executable(test_stereo test_stereo.cpp)
target_link_libraries(
  test_stereo PRIVATE hwcv opencv_core opencv_highgui opencv_imgproc
                      Boost::filesystem spdlog::spdlog)

install(TARGETS test_cvt_color test_stereo RUNTIME DESTINATION lib/hwcv)

install(DIRECTORY data DESTINATION lib/hwcv/)
