#pragma once

#include <boost/core/noncopyable.hpp>
#include <opencv2/imgproc.hpp>

namespace hwcv {

/**
 * @brief Color conversion Interface
 *
 */
class ICvtColor {
 public:
  /**
   * @brief Execute color conversion
   *
   * @param src Input image
   * @param dst Output image
   * @param code Color conversion code
   */
  virtual void Execute(cv::Mat const& src, cv::Mat& dst,
                       cv::ColorConversionCodes code) = 0;
  /**
   * @brief Destroy the ICvtColor object
   *
   */
  virtual ~ICvtColor(){};
};

class CvtColor : public ICvtColor {
 public:
  /**
   * @brief Construct a new color conversion object. The binary file is
   * automatically loaded
   *
   */
  CvtColor();
  ~CvtColor() = default;

  CvtColor(CvtColor&&) = delete;
  CvtColor& operator=(CvtColor&&) = delete;
  CvtColor(CvtColor const&) = delete;
  CvtColor& operator=(CvtColor const&) = delete;
  void Execute(cv::Mat const& src, cv::Mat& dst,
               cv::ColorConversionCodes code) override;

 private:
  std::unique_ptr<ICvtColor> impl_;
};

}  // namespace hwcv