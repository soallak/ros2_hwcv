#include <boost/filesystem.hpp>
#include <hwcv/stereo.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "utils.hpp"

namespace fs = boost::filesystem;

void TestStereo(fs::path left_path, fs::path right_path) {
  spdlog::debug("Load images");
  auto left_src =
      cv::imread(left_path.string(), cv::ImreadModes::IMREAD_GRAYSCALE);
  auto right_src =
      cv::imread(right_path.string(), cv::ImreadModes::IMREAD_GRAYSCALE);

  spdlog::debug("Execute Stereo matching");
  hwcv::StereoMatcher matcher;
  cv::Mat dst(left_src.rows, left_src.cols, CV_16UC1);
  matcher.Execute(left_src, right_src, dst);

  // The hw implementation uses 12.4 fixed point
  // convert to fload then / 16 then normalize
  cv::Mat float_dst;
  cv::Mat float_norm_dst;
  dst.convertTo(float_dst, CV_32F, 1.0 / 16.f);
  cv::normalize(float_dst, float_norm_dst);
  cv::imwrite("test_stereo.png", float_dst);
}

int main(int argc, char** argv) {
  fs::path prog_dir = fs::path(argv[0]).parent_path();

  auto left_path = fs::path(prog_dir).append("data/test_stereo_left.png");
  auto right_path = fs::path(prog_dir).append("data/test_stereo_right.png");

  TRACE(TestStereo(left_path, right_path));
}