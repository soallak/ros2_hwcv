#include <spdlog/fmt/fmt.h>

#include <hwcv/stereo.hpp>
#include <kernels/stereo/accel.hpp>
#include <vitis_common/xcl2/xcl2.hpp>

#include "cl_utils.hpp"
#include "except.hpp"
#include "stereo_lbm_config.hpp"
#include "utils.hpp"
#include "vitis_algorithm.hpp"

namespace hwcv {

class StereoImpl : public IStereoMatcher, public VitisAlgorithm {
 public:
  StereoImpl() {
    try {
      LoadBinaryFile();
      logger_->info(fmt::format(
          FMT_STRING("Load binary file {} for device {} succeeded."),
          GetBinaryFile(), device_.getInfo<CL_DEVICE_NAME>()));
      LoadKernel();
    } catch (CLError const &e) {
      logger_->error(fmt::format(FMT_STRING("Load binary file {} failed: {}"),
                                 GetBinaryFile(), e.what()));
      is_binary_loaded_ = false;
    }
  }
  ~StereoImpl() {}

  StereoImpl(StereoImpl &&) = delete;
  StereoImpl &operator=(StereoImpl &&) = delete;
  StereoImpl(StereoImpl const &) = delete;
  StereoImpl &operator=(StereoImpl const &) = delete;

  void Execute(cv::Mat const &left_src, cv::Mat const &right_src,
               cv::Mat &dst) override {
    ValidateArguments(left_src, right_src, dst);
    if (is_binary_loaded_) {
      logger_->debug("Create Buffers");
      cl::Buffer buffer_left_src =
          CreateBufferForCvMat(left_src, context_, CL_MEM_READ_ONLY);
      cl::Buffer buffer_right_src =
          CreateBufferForCvMat(right_src, context_, CL_MEM_READ_ONLY);
      cl::Buffer buffer_dst =
          CreateBufferForCvMat(dst, context_, CL_MEM_WRITE_ONLY);

      int height = left_src.rows;
      int width = left_src.cols;

      logger_->debug("Setup kernel arguments");
      SetKernelArg(kernel_, 0, buffer_left_src, buffer_right_src, buffer_dst,
                   height, width, pre_filter_cap_, min_disparity_,
                   uniqueness_ratio_, texture_threshold_);

      logger_->debug("Write input buffers");
      cl_int err = 0;
      OCL_CALL(err = queue_.enqueueWriteBuffer(buffer_left_src, CL_TRUE, 0,
                                               GetBufferSizeForCvMat(left_src),
                                               left_src.data),
               err);
      OCL_CALL(err = queue_.enqueueWriteBuffer(buffer_right_src, CL_TRUE, 0,
                                               GetBufferSizeForCvMat(right_src),
                                               right_src.data),
               err);

      logger_->debug("Execute stereo kernel");
      OCL_CALL(err = queue_.enqueueTask(kernel_), err);

      logger_->debug("Read output buffers");
      OCL_CALL(
          err = queue_.enqueueReadBuffer(buffer_dst, CL_TRUE, 0,
                                         GetBufferSizeForCvMat(dst), dst.data),
          err);

      logger_->debug("Finish up queue");
      OCL_CALL(err = queue_.finish(), err);

    } else {
      logger_->warn(fmt::format(
          FMT_STRING(
              "Binary file is not loaded. {} will use software emulation"),
          __func__));

      logger_->debug("Create xf matrices");

      auto xf_left_src_mat =
          CvMat2XfMat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1>(left_src);
      auto xf_right_src_mat =
          CvMat2XfMat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1>(right_src);

      xf::cv::Mat<XF_16UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> xf_dst_mat(
          dst.rows, dst.cols, static_cast<void *>(dst.data));

      logger_->debug("Create ap_uint buffers");
      auto xf_left_src_ptr =
          CreateApUintData<XF_PTR_SRC_WIDTH>(xf_left_src_mat);
      auto xf_right_src_ptr =
          CreateApUintData<XF_PTR_SRC_WIDTH>(xf_right_src_mat);
      ap_uint<XF_PTR_DST_WIDTH> *xf_dst_ptr =
          CreateApUintData<XF_PTR_DST_WIDTH, XF_16UC1, XF_HEIGHT, XF_WIDTH,
                           XF_NPPC1>();

      logger_->debug("Execute software emulation of stereo_lbm_accel");
      stereo_lbm_accel(xf_left_src_ptr, xf_right_src_ptr, xf_dst_ptr,
                       left_src.rows, left_src.cols, pre_filter_cap_,
                       min_disparity_, uniqueness_ratio_, texture_threshold_);

      logger_->debug("Copy output to dst");
      xf::cv::Array2xfMat(xf_dst_ptr, xf_dst_mat);

      logger_->debug("Destroy buffers");
      DestroyApUint(xf_left_src_ptr);
      DestroyApUint(xf_right_src_ptr);
      DestroyApUint(xf_dst_ptr);
    }
  }

  void SetPreFilterCapSize(int val) override {
    constexpr int max = 63;
    constexpr int min = 1;
    if (!InRange<int>::In<min, max>(val)) {
      throw InvalidArgument(fmt::format(
          FMT_STRING("Pre Filter Cap Size {} not in range [{}, {}]"), val, min,
          max));
    }
  }

  void SetMinDisparity(int val) override {
    constexpr int max = hwcv::kernels::stereo::lbm::GetMaxDisparity();
    constexpr int min = 0;
    if (!InRange<int>::In<min, max>(val)) {
      throw InvalidArgument(fmt::format(
          FMT_STRING("Min Disparity {} not in range [{}, {}]"), val, min, max));
    }
  }

  void SetUniquenessRatio(int val) override {
    constexpr int max = 100;
    constexpr int min = 0;
    if (!InRange<int>::In<min, max>(val)) {
      throw InvalidArgument(
          fmt::format(FMT_STRING("Uniqueness Ratio {} not in range [{}, {}]"),
                      val, min, max));
    }
  }

  void SetTextureThreshold(int val) override {
    constexpr int min = 0;
    if (val < min) {
      throw InvalidArgument(
          fmt::format(FMT_STRING("Texture Threshold is not positive")));
    }
  }

  inline int GetPreFilterCapSize() override { return pre_filter_cap_; }

  inline int GetMinDisparity() override { return min_disparity_; }

  inline int GetUniquenessRatio() override { return uniqueness_ratio_; }

  inline int GetTextureThreshold() override { return texture_threshold_; }

  std::string GetBinaryFile() override {
    return hwcv::kernels::stereo::GetBinaryFile();
  }

 private:
  template <int HEIGHT, int WIDTH>
  bool IsValidSize(cv::Mat const &mat) {
    return InRange<int>::In<0, HEIGHT>(mat.rows) &&
           InRange<int>::In<0, WIDTH>(mat.cols);
  }

  void ValidateArguments(cv::Mat const &left_src, cv::Mat const &right_src,
                         cv::Mat const &dst) {
    // validate sizes
    if (!IsValidSize<XF_HEIGHT, XF_WIDTH>(left_src)) {
      throw InvalidArgument(fmt::format(
          FMT_STRING(
              "left image size ({}, {}) exceeds the maximum size ({}, {})"),
          left_src.rows, left_src.cols, XF_HEIGHT, XF_WIDTH));
    }
    if (!IsValidSize<XF_HEIGHT, XF_WIDTH>(right_src)) {
      throw InvalidArgument(fmt::format(
          FMT_STRING(
              "left image size ({}, {}) exceeds the maximum size ({}, {})"),
          right_src.rows, right_src.cols, XF_HEIGHT, XF_WIDTH));
    }
    if (!IsValidSize<XF_HEIGHT, XF_WIDTH>(dst)) {
      throw InvalidArgument(fmt::format(
          FMT_STRING(
              "left image size ({}, {}) exceeds the maximum size ({}, {})"),
          dst.rows, dst.cols, XF_HEIGHT, XF_WIDTH));
    }

    // then types

    if (left_src.type() != CV_8UC1) {
      throw InvalidArgument(fmt::format("left image type is {} but must be {}",
                                        left_src.type(), CV_8UC1));
    }
    if (right_src.type() != CV_8UC1) {
      throw InvalidArgument(fmt::format("right image type is {} but must be {}",
                                        right_src.type(), CV_8UC1));
    }
    if (dst.elemSize() != 2) {
      throw InvalidArgument(
          fmt::format("dst image elemSize must be equal to 2"));
    }
  }

  void LoadKernel() {
    std::string kernel_name = hwcv::kernels::stereo::lbm::GetName();
    cl_int err = 0;
    logger_->debug(fmt::format(FMT_STRING("Load kernel {}"), kernel_name));
    OCL_CALL(kernel_ = cl::Kernel(program_, kernel_name.c_str(), &err), err);
  }

 private:
  // TODO(soallak) define consts for these defaults
  int pre_filter_cap_{31};
  int min_disparity_{0};
  int uniqueness_ratio_{15};
  int texture_threshold_{10};
  cl::Kernel kernel_;
};

StereoMatcher::StereoMatcher() : impl_(std::make_unique<StereoImpl>()) {}

void StereoMatcher::Execute(cv::Mat const &left_src, cv::Mat const &right_src,
                            cv::Mat &dst) {
  impl_->Execute(left_src, right_src, dst);
}

void StereoMatcher::SetPreFilterCapSize(int val) {
  impl_->SetPreFilterCapSize(val);
}

void StereoMatcher::SetMinDisparity(int val) { impl_->SetMinDisparity(val); }

void StereoMatcher::SetUniquenessRatio(int val) {
  impl_->SetUniquenessRatio(val);
}

void StereoMatcher::SetTextureThreshold(int val) {
  impl_->SetTextureThreshold(val);
}

inline int StereoMatcher::GetPreFilterCapSize() {
  return impl_->GetPreFilterCapSize();
}

inline int StereoMatcher::GetMinDisparity() { return impl_->GetMinDisparity(); }

inline int StereoMatcher::GetUniquenessRatio() {
  return impl_->GetUniquenessRatio();
}

inline int StereoMatcher::GetTextureThreshold() {
  return impl_->GetTextureThreshold();
}

}  // namespace hwcv