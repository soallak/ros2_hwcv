#include <spdlog/spdlog.h>

#include <boost/filesystem.hpp>
#include <hwcv/feature.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include "utils.hpp"

namespace fs = boost::filesystem;

void TestFastFeatureDetector(fs::path const& img_path) {
  spdlog::debug(fmt::format("Load {} image", img_path.string()));
  auto img = cv::imread(img_path.string(), cv::IMREAD_GRAYSCALE);

  spdlog::debug("Execute Fast Feature Detector");
  hwcv::FastFeatureDetector fast;
  std::vector<cv::KeyPoint> keypoints;
  fast.Execute(img, keypoints);

  spdlog::debug(
      fmt::format(FMT_STRING("Draw {} keypoints on image"), keypoints.size()));
  cv::Mat dst;
  cv::drawKeypoints(img, keypoints, dst);

  std::string write_img_path("test_feature_fast.png");
  spdlog::debug(fmt::format("Save {} image", write_img_path));

  cv::imwrite(write_img_path, dst);
}

int main(int /* argc */, char** argv) {
  spdlog::set_level(spdlog::level::trace);

  fs::path img_path = fs::path(std::string(argv[0]))
                          .parent_path()
                          .append("data/test_feature.png");
  TRACE(TestFastFeatureDetector(img_path));
}