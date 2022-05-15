#include <hwcv/feature.hpp>
#include <kernels/feature/accel.hpp>
#include <mutex>

#include "cl_utils.hpp"
#include "utils.hpp"
#include "vitis_algorithm.hpp"

namespace fast = hwcv::kernels::feature::fast;

namespace hwcv {

class FastFeatureDetectorImpl : public IFastFeatureDetector, VitisAlgorithm {
 public:
  FastFeatureDetectorImpl() {
    try {
      LoadBinaryFile();
      logger_->info(fmt::format(
          FMT_STRING("Load binary file {} for device {} succeeded."),
          GetBinaryFile(), device_.getInfo<CL_DEVICE_NAME>()));
      LoadKernel();
      SetupBuffers();
    } catch (CLError const& e) {
      logger_->error(fmt::format(FMT_STRING("Load binary file {} failed: {}"),
                                 GetBinaryFile(), e.what()));
      is_binary_loaded_ = false;
    }
  }

  void Execute(cv::Mat const& src,
               std::vector<cv::KeyPoint>& keypoints) override {
    ValidateArguments(src);
    cv::Mat dst(src.rows, src.cols, CV_8UC1);
    if (is_binary_loaded_) {
      HWExecute(src, dst);
    } else {
      logger_->warn(fmt::format(
          FMT_STRING(
              "Binary file is not loaded. {} will use software emulation"),
          __func__));
      SWExecute(src, dst);
    }
    logger_->debug("Convert to cv::Keypoint");
    ToKeypoints(dst, keypoints);
  }

  inline std::string GetBinaryFile() override {
    return hwcv::kernels::feature::GetBinaryFile();
  }

  void SetThreshold(int threshold) override {
    constexpr auto min = 0;
    constexpr auto max = 255;
    if (!InRange<int>::In<min, max>(threshold)) {
      throw InvalidArgument(fmt::format(
          FMT_STRING("Threshold ({}) must be in the range [{}, {}]"), threshold,
          min, max));
    } else {
      threshold_ = threshold;
    }
  }
  inline int GetThreshold() override { return threshold_; }

 private:
  void LoadKernel() {
    auto const kernel_name = hwcv::kernels::feature::fast::GetName();
    cl_int err = 0;
    logger_->debug(fmt::format(FMT_STRING("Load kernel {}"), kernel_name));
    OCL_CALL(kernel_ = cl::Kernel(program_, kernel_name.c_str(), &err), err);
  }

  void ValidateArguments(cv::Mat const& mat) {
    if (!IsValidSize<fast::GetMaxHeight(), fast::GetMaxWidth()>(mat)) {
      throw InvalidArgument(fmt::format(
          FMT_STRING("image size ({}, {}) exceeds the maximum size ({}, {})"),
          mat.rows, mat.cols, fast::GetMaxHeight(), fast::GetMaxWidth()));
    }
    if (mat.type() != CV_8UC1) {
      throw InvalidArgument(fmt::format(
          FMT_STRING("Matrice type ({}) must be {}"), mat.type(), CV_8UC1));
    }
  }
  template <int HEIGHT, int WIDTH>
  bool IsValidSize(cv::Mat const& mat) {
    return InRange<int>::In<0, HEIGHT>(mat.rows) &&
           InRange<int>::In<0, WIDTH>(mat.cols);
  }

  // TODO(soallak): Is it possible to factorize the HW Execution code ?
  void HWExecute(cv::Mat const& src, cv::Mat& dst) {
    int height = src.rows;
    int width = src.cols;

    logger_->debug("Setup fast arguments");
    SetKernelArg(kernel_, 0, buffer_src_, buffer_dst_, height, width,
                 threshold_);

    logger_->debug("Write input buffers");
    cl_int err = 0;
    cl::Event event;
    OCL_CALL(err = queue_.enqueueWriteBuffer(buffer_src_, CL_TRUE, 0,
                                             GetBufferSizeForCvMat(src),
                                             src.data, nullptr, &event),
             err);

    logger_->debug("Execute stereo kernel");
    OCL_CALL(err = queue_.enqueueTask(kernel_), err);

    logger_->debug("Read output buffers");
    OCL_CALL(err = queue_.enqueueReadBuffer(buffer_dst_, CL_TRUE, 0,
                                            GetBufferSizeForCvMat(dst),
                                            dst.data, nullptr, &event),
             err);

    logger_->debug("Finish up queue");
    OCL_CALL(err = queue_.flush(), err);
  }

  void SWExecute(cv::Mat const& src, cv::Mat& dst) {
    logger_->debug("Create xf matrices");

    auto xf_src_mat = CvMat2XfMat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1>(src);
    xf::cv::Mat<XF_16UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> xf_dst_mat(
        dst.rows, dst.cols, static_cast<void*>(dst.data));

    logger_->debug("Create ap_uint buffers");
    auto xf_src_ptr = CreateApUintData<XF_PTR_SRC_WIDTH>(xf_src_mat);
    ap_uint<XF_PTR_DST_WIDTH>* xf_dst_ptr =
        CreateApUintData<XF_PTR_DST_WIDTH, XF_8UC1, XF_HEIGHT, XF_WIDTH,
                         XF_NPPC1>();

    logger_->debug("Execute software emulation of fast_accel");
    fast_accel(xf_src_ptr, xf_dst_ptr, src.rows, src.cols, threshold_);

    logger_->debug("Copy output to dst");
    xf::cv::Array2xfMat(xf_dst_ptr, xf_dst_mat);

    logger_->debug("Destroy buffers");
    DestroyApUint(xf_src_ptr);
    DestroyApUint(xf_dst_ptr);
  }

  void ToKeypoints(cv::Mat& mat, std::vector<cv::KeyPoint>& keypoints) {
    keypoints.clear();
    keypoints.reserve(mat.rows * mat.cols);

    std::mutex mtx;

    mat.forEach<unsigned char>(
        [&keypoints, &mtx](unsigned char& pixel, const int* pos) {
          cv::KeyPoint kp(static_cast<float>(pos[0]),
                          static_cast<float>(pos[1]), pixel);
          kp.response = pixel;

          if (pixel > 0) {
            std::lock_guard<std::mutex> lk(mtx);
            keypoints.push_back(kp);
          }
        });
  }

  void SetupBuffers() {
    logger_->debug("Create Buffers");
    cl_int err(0);
    OCL_CALL(buffer_src_ = cl::Buffer(context_, CL_MEM_READ_ONLY,
                                      buffer_src_size_, NULL, &err),
             err)
    OCL_CALL(buffer_dst_ = cl::Buffer(context_, CL_MEM_WRITE_ONLY,
                                      buffer_dst_size_, NULL, &err),
             err)
  }

 private:
  constexpr static int default_threshold_ = 20;
  int threshold_{default_threshold_};
  cl::Kernel kernel_;
  cl::Buffer buffer_src_;
  cl::Buffer buffer_dst_;
  const unsigned int buffer_src_size_ =
      fast::GetMaxHeight() * fast::GetMaxWidth();
  const unsigned int buffer_dst_size_ =
      fast::GetMaxHeight() * fast::GetMaxWidth();
};

FastFeatureDetector::FastFeatureDetector()
    : impl_(std::make_unique<FastFeatureDetectorImpl>()) {}

void FastFeatureDetector::Execute(const cv::Mat& src,
                                  std::vector<cv::KeyPoint>& keypoints) {
  impl_->Execute(src, keypoints);
}

inline void FastFeatureDetector::SetThreshold(int threshold) {
  impl_->SetThreshold(threshold);
}

inline int FastFeatureDetector::GetThreshold() { return impl_->GetThreshold(); }

}  // namespace hwcv