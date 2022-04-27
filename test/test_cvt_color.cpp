#include <spdlog/spdlog.h>

#include <boost/filesystem.hpp>
#include <hwcv/cvt_color.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "utils.hpp"

namespace fs = boost::filesystem;

void TestDemosaicing(fs::path img_path) {
  spdlog::debug(fmt::format("Load {} image", img_path.string()));
  auto src = cv::imread(img_path.string(), cv::IMREAD_GRAYSCALE);
  cv::Mat dst(src.rows, src.cols, CV_8UC4);

  hwcv::CvtColor cvt_color;
  cvt_color.Execute(src, dst, cv::ColorConversionCodes::COLOR_BayerBG2RGBA);

  std::string write_img("test_cvt_color_demosaicing.png");
  spdlog::debug(fmt::format("Save image {}", write_img));
  cv::imwrite(write_img, dst);
}

void TestBGR2Gray(fs::path img_path) {
  spdlog::debug(fmt::format("Load {} image", img_path.string()));
  auto src = cv::imread(img_path.string());
  cv::Mat dst(src.rows, src.cols, CV_8UC1);

  hwcv::CvtColor cvt_color;
  cvt_color.Execute(src, dst, cv::ColorConversionCodes::COLOR_BGR2GRAY);

  std::string write_img("test_cvt_color_bgr2gray.png");
  spdlog::debug(fmt::format("Save image {}", write_img));
  cv::imwrite(write_img, dst);
}

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::trace);

  fs::path raw_img_path = fs::path(std::string(argv[0]))
                              .parent_path()
                              .append("data/test_cvt_color_color.jpg");

  fs::path color_img_path = fs::path(std::string(argv[0]))
                                .parent_path()
                                .append("data/test_cvt_color_color.jpg");

  TRACE(TestDemosaicing(raw_img_path));
  TRACE(TestBGR2Gray(color_img_path));
}
