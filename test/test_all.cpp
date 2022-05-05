#include <spdlog/spdlog.h>

#include <boost/filesystem.hpp>
#include <hwcv/cvt_color.hpp>
#include <hwcv/feature.hpp>
#include <hwcv/stereo.hpp>

#include "utils.hpp"

namespace fs = boost::filesystem;

void TestMultiThreaded() {
  // TODO(soallak): Implement
}

void TestSingleThreaded() {
  constexpr int rows = 768;
  constexpr int cols = 1024;
  using namespace hwcv;
  {
    CvtColor cvt_color;
    cv::Mat src(rows, cols, CV_8UC3);
    cv::Mat dst(rows, cols, CV_8UC1);

    TRACE(BENCHMARK(
        cvt_color.Execute(src, dst, cv::ColorConversionCodes::COLOR_BGR2GRAY)));
  }
  {
    StereoMatcher stereo;
    cv::Mat src_l(rows, cols, CV_8UC1);
    cv::Mat src_r(rows, cols, CV_8UC1);

    cv::Mat dst(rows, cols, CV_16UC1);

    TRACE(BENCHMARK(stereo.Execute(src_l, src_r, dst)));
  }
  {
    cv::Mat src(rows, cols, CV_8UC1);
    std::vector<cv::KeyPoint> keypoints;
    FastFeatureDetector fast;

    TRACE(BENCHMARK(fast.Execute(src, keypoints)));
  }
}

int main(int /* argc */, char** /* argv */) {
  spdlog::set_level(spdlog::level::trace);
  TRACE(TestSingleThreaded());
}