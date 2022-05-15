#pragma once

#include <opencv2/core.hpp>

namespace hwcv {

class IStereoMatcher {
 public:
  /**
   * @brief Compute disparities for stereo pair
   *
   * @param left_src Left input image
   * @param right_src Right input image
   * @param dst Computed disparity image
   */
  virtual void Execute(cv::Mat const& left_src, cv::Mat const& right_src,
                       cv::Mat& dst) = 0;

  virtual void SetPreFilterCapSize(int val) = 0;

  virtual void SetMinDisparity(int val) = 0;

  virtual void SetUniquenessRatio(int val) = 0;

  virtual void SetTextureThreshold(int val) = 0;

  virtual int GetPreFilterCapSize() const = 0;

  virtual int GetMinDisparity() const = 0;

  virtual int GetUniquenessRatio() const = 0;

  virtual int GetTextureThreshold() const = 0;

  virtual int GetNumDisparities() const = 0;

  virtual int GetAggregationWindowSize() const = 0;

  /**
   * @brief Destroy the IStereoMatcher object
   *
   */
  virtual ~IStereoMatcher() {}
};

class StereoMatcher : public IStereoMatcher {
 public:
  /**
   * @brief Construct a new Stereo Matcher object. The binary file is
   * automatically loaded
   *
   */
  StereoMatcher();
  ~StereoMatcher() = default;

  StereoMatcher(StereoMatcher&&) = delete;
  StereoMatcher& operator=(StereoMatcher&&) = delete;
  StereoMatcher(StereoMatcher const&) = delete;
  StereoMatcher& operator=(StereoMatcher const&) = delete;

  void Execute(cv::Mat const& left_src, cv::Mat const& right_src,
               cv::Mat& dst) override;

  void SetPreFilterCapSize(int val) override;

  void SetMinDisparity(int val) override;

  void SetUniquenessRatio(int val) override;

  void SetTextureThreshold(int val) override;

  int GetPreFilterCapSize() const override;

  int GetMinDisparity() const override;

  int GetUniquenessRatio() const override;

  int GetTextureThreshold() const override;

  int GetNumDisparities() const override;

  int GetAggregationWindowSize() const override;

 private:
  std::unique_ptr<IStereoMatcher> impl_;
};

}  // namespace hwcv