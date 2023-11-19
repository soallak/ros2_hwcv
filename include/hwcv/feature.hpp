#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <memory>

namespace hwcv {

class IFastFeatureDetector {
 public:
  /**
   * @brief
   *
   * @param src Input image
   * @param keypoints Detected keypoints
   */
  virtual void Execute(cv::Mat const& src,
                       std::vector<cv::KeyPoint>& keypoints) = 0;
  /**
   * @brief Set the different in pixel intensity threshold
   *
   * @param threshold threshold value
   */
  virtual void SetThreshold(int threshold) = 0;

  /**
   * @brief Get the threshold
   *
   * @return int the threshold value
   */
  virtual int GetThreshold() = 0;

  /**
   * @brief Destroy the IFeature object
   *
   */
  virtual ~IFastFeatureDetector() {}
};

class FastFeatureDetector : public IFastFeatureDetector {
 public:
  FastFeatureDetector();
  ~FastFeatureDetector() = default;

  FastFeatureDetector(FastFeatureDetector&&) = delete;
  FastFeatureDetector& operator=(FastFeatureDetector&&) = delete;
  FastFeatureDetector(FastFeatureDetector const&) = delete;
  FastFeatureDetector& operator=(FastFeatureDetector const&) = delete;

  void Execute(cv::Mat const& src,
               std::vector<cv::KeyPoint>& keypoints) override;

  void SetThreshold(int threshold) override;
  int GetThreshold() override;

 private:
  std::unique_ptr<IFastFeatureDetector> impl_;
};

}  // namespace hwcv
