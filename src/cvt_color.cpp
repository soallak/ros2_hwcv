

#include <hwcv/cvt_color.hpp>
#include <kernels/cvt_color/accel.hpp>

#include "cl_utils.hpp"
#include "cvt_color_config.hpp"
#include "except.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include "vitis_algorithm.hpp"

namespace demosaicing = hwcv::kernels::cvt_color::demosaicing;
namespace bgr2gray = hwcv::kernels::cvt_color::bgr2gray;

namespace hwcv {

class CvtColorImpl : public VitisAlgorithm, public ICvtColor {
 public:
  CvtColorImpl() {
    try {
      LoadBinaryFile();
      logger_->info(fmt::format(
          FMT_STRING("Load binary file {} for device {} succeeded."),
          GetBinaryFile(), device_.getInfo<CL_DEVICE_NAME>()));
      LoadKernels();
      SetupBuffers();
    } catch (CLError const &e) {
      logger_->error(fmt::format(FMT_STRING("Load binary file {} failed: {}"),
                                 GetBinaryFile(), e.what()));
      is_binary_loaded_ = false;
    }
  }

  ~CvtColorImpl() {}

  CvtColorImpl(CvtColorImpl &&) = delete;
  CvtColorImpl &operator=(CvtColorImpl &&) = delete;
  CvtColorImpl(CvtColorImpl const &) = delete;
  CvtColorImpl &operator=(CvtColorImpl const &) = delete;

  std::string GetBinaryFile() override {
    return hwcv::kernels::cvt_color::GetBinaryFile();
  }

  void Execute(cv::Mat const &src, cv::Mat &dst,
               cv::ColorConversionCodes code) override {
    ValidateArguments(src, dst, code);
    if (is_binary_loaded_) {
      int height = src.rows;
      int width = src.cols;

      cl::Kernel kernel;
      if (code == cv::ColorConversionCodes::COLOR_BGR2GRAY) {
        logger_->debug("Setup bgr2gray_kernel arguments");

        kernel = bgr2gray_kernel_;
        SetKernelArg(kernel, 0, buffer_src_, buffer_dst_, height, width);

      } else {
        logger_->debug("Setup demosaicing_kernel arguments");
        kernel = demosaicing_kernel_;
        auto pattern = ToXfPattern(code);
        SetKernelArg(kernel, 0, buffer_src_, buffer_dst_, height, width,
                     pattern);
      }
      logger_->debug("Write input buffers");
      cl_int err = 0;
      cl::Event event;
      OCL_CALL(err = queue_.enqueueWriteBuffer(buffer_src_, CL_TRUE, 0,
                                               GetBufferSizeForCvMat(src),
                                               src.data, nullptr, &event),
               err);

      logger_->debug("Execute stereo kernel");
      OCL_CALL(err = queue_.enqueueTask(kernel), err);

      logger_->debug("Read output buffers");
      OCL_CALL(err = queue_.enqueueReadBuffer(buffer_dst_, CL_TRUE, 0,
                                              GetBufferSizeForCvMat(dst),
                                              dst.data, nullptr, &event),
               err);

      logger_->debug("Finish up queue");
      OCL_CALL(err = queue_.flush(), err);
    }

    else {
      // TODO(soallak): protects this parts with macros
      logger_->warn(fmt::format(
          FMT_STRING(
              "Binary file is not loaded. {} will use software emulation"),
          __func__));

      XF_demosaicing pattern(XF_BAYER_RG);
      // TODO(soallak) This is ugly. Better to use conversion functions
      switch (code) {
        case cv::ColorConversionCodes::COLOR_BayerBG2RGBA:
        case cv::ColorConversionCodes::COLOR_BayerGB2RGBA:
        case cv::ColorConversionCodes::COLOR_BayerGR2RGBA:
        case cv::ColorConversionCodes::COLOR_BayerRG2RGBA: {
          logger_->debug("Create xf matrices");
          pattern = ToXfPattern(code);
          auto xf_src_mat =
              CvMat2XfMat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1>(src);
          xf::cv::Mat<XF_8UC4, XF_HEIGHT, XF_WIDTH, XF_NPPC1> xf_dst_mat(
              dst.rows, dst.cols, static_cast<void *>(dst.data));

          logger_->debug("Create ap_uint buffers");
          ap_uint<XF_PTR_SRC_WIDTH> *xf_src_ptr =
              CreateApUintData<XF_PTR_SRC_WIDTH>(xf_src_mat);
          ap_uint<XF_PTR_SRC_WIDTH> *xf_dst_ptr =
              CreateApUintData<XF_PTR_DST_WIDTH, XF_8UC4, XF_HEIGHT, XF_WIDTH,
                               XF_NPPC1>();
          logger_->debug(fmt::format(
              FMT_STRING("Simulate demosaicing_accel on an {}x{} image"),
              xf_src_mat.rows, xf_src_mat.cols));
          demosaicing_accel(xf_src_ptr, xf_dst_ptr, xf_src_mat.rows,
                            xf_src_mat.cols, pattern);
          logger_->debug("Copy result from buffer");
          xf::cv::Array2xfMat(xf_dst_ptr, xf_dst_mat);

          logger_->debug("Destroy ap_uint buffers");
          DestroyApUint(xf_src_ptr);
          DestroyApUint(xf_dst_ptr);
          logger_->debug(fmt::format(FMT_STRING("Exit {}"), __func__));
          break;
        }

        case cv::ColorConversionCodes::COLOR_BGR2GRAY: {
          logger_->debug("Create xf matrices");

          auto xf_src_mat =
              CvMat2XfMat<XF_8UC3, XF_HEIGHT, XF_WIDTH, XF_NPPC1>(src);
          xf::cv::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> xf_dst_mat(
              dst.rows, dst.cols, static_cast<void *>(dst.data));

          logger_->debug("Create ap_uint buffers");
          ap_uint<XF_PTR_SRC_WIDTH> *xf_src_ptr =
              CreateApUintData<XF_PTR_SRC_WIDTH>(xf_src_mat);
          ap_uint<XF_PTR_SRC_WIDTH> *xf_dst_ptr =
              CreateApUintData<XF_PTR_DST_WIDTH, XF_8UC1, XF_HEIGHT, XF_WIDTH,
                               XF_NPPC1>();
          logger_->debug(
              fmt::format(FMT_STRING("Simulate bgr2_accel on an {}x{} image"),
                          xf_src_mat.rows, xf_src_mat.cols));

          bgr2gray_accel(xf_src_ptr, xf_dst_ptr, xf_src_mat.rows,
                         xf_src_mat.cols);

          logger_->debug("Copy result from buffer");
          xf::cv::Array2xfMat(xf_dst_ptr, xf_dst_mat);

          logger_->debug("Destroy ap_uint buffer");
          DestroyApUint(xf_src_ptr);
          DestroyApUint(xf_dst_ptr);
          break;
        }

        default:
          // conversion code validation
          throw InvalidArgument(fmt::format(
              FMT_STRING("Color conversion code {} is not supported"), code));
      }
    }
  }

 private:
  void ValidateArguments(cv::Mat const &src, cv::Mat const &dst,
                         cv::ColorConversionCodes code) {
    int src_required_type = 0;
    int dst_required_type = 0;
    size_t max_width = 0;
    size_t max_height = 0;
    switch (code) {
      case cv::ColorConversionCodes::COLOR_BayerBG2RGBA:
      case cv::ColorConversionCodes::COLOR_BayerGB2RGBA:
      case cv::ColorConversionCodes::COLOR_BayerGR2RGBA:
      case cv::ColorConversionCodes::COLOR_BayerRG2RGBA: {
        src_required_type = CV_8UC1;
        dst_required_type = CV_8UC4;
        max_width = kernels::cvt_color::demosaicing::GetMaxWidth();
        max_height = kernels::cvt_color::demosaicing::GetMaxHeight();
        break;
      }

      case cv::ColorConversionCodes::COLOR_BGR2GRAY: {
        src_required_type = CV_8UC3;
        dst_required_type = CV_8UC1;
        max_width = kernels::cvt_color::bgr2gray::GetMaxWidth();
        max_height = kernels::cvt_color::bgr2gray::GetMaxHeight();
        break;
      }

      default:
        // conversion code validation
        throw InvalidArgument(fmt::format(
            FMT_STRING("Color conversion code {} is not supported"), code));
    }

    auto validate = [max_height, max_width](cv::Mat const &mat,
                                            int required_type,
                                            std::string tag) {
      if (mat.type() != required_type) {
        throw InvalidArgument(fmt::format(
            FMT_STRING("{} image required type {} is not equal to {}"), tag,
            mat.type(), required_type));
      }

      auto const &rows = mat.rows;
      auto const &cols = mat.cols;
      if (mat.rows > static_cast<int>(max_height) ||
          mat.cols > static_cast<int>(max_width) || mat.rows < 1 ||
          mat.cols < 1) {
        throw InvalidArgument(fmt::format(
            FMT_STRING(
                "{} image size ({}, {}) is not within maximum size ({}, {})"),
            tag, rows, cols, max_height, max_width));
      }
    };

    // images validation
    validate(src, src_required_type, "src");
    validate(dst, dst_required_type, "dst");
  }
  void LoadDemosaicingKernel() {
    std::string kernel_name = hwcv::kernels::cvt_color::demosaicing::GetName();
    cl_int err = 0;
    logger_->debug(fmt::format(FMT_STRING("Load kernel {}"), kernel_name));
    OCL_CALL(
        demosaicing_kernel_ = cl::Kernel(program_, kernel_name.c_str(), &err),
        err);
  }
  void LoadBgr2GrayKernel() {
    std::string kernel_name = hwcv::kernels::cvt_color::bgr2gray::GetName();
    cl_int err = 0;
    logger_->debug(fmt::format(FMT_STRING("Load kernel {}"), kernel_name));
    OCL_CALL(bgr2gray_kernel_ = cl::Kernel(program_, kernel_name.c_str(), &err),
             err);
  }

  void LoadKernels() {
    LoadBgr2GrayKernel();
    LoadDemosaicingKernel();
  }

  static XF_demosaicing ToXfPattern(cv::ColorConversionCodes code) {
    switch (code) {
      case cv::ColorConversionCodes::COLOR_BayerBG2RGBA: {
        return XF_BAYER_BG;
      }
      case cv::ColorConversionCodes::COLOR_BayerGB2RGBA: {
        return XF_BAYER_GB;
      }
      case cv::ColorConversionCodes::COLOR_BayerGR2RGBA: {
        return XF_BAYER_GR;
      }
      case cv::ColorConversionCodes::COLOR_BayerRG2RGBA: {
        return XF_BAYER_RG;
      }

      default:
        // conversion code validation
        throw InvalidArgument(fmt::format(
            FMT_STRING("Color conversion code {} is not supported"), code));
    }
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
  cl::Kernel demosaicing_kernel_;
  cl::Kernel bgr2gray_kernel_;
  cl::Buffer buffer_dst_;
  cl::Buffer buffer_src_;
  const unsigned long max_width_ =
      std::max(demosaicing::GetMaxWidth(), bgr2gray::GetMaxWidth());
  const unsigned long max_height_ =
      std::max(demosaicing::GetMaxHeight(), bgr2gray::GetMaxHeight());
  const unsigned long buffer_dst_size_ = max_height_ * max_width_ * 3;
  const unsigned long buffer_src_size_ = max_height_ * max_width_ * 4;
};

CvtColor::CvtColor() : impl_(std::make_unique<CvtColorImpl>()) {}

void CvtColor::Execute(const cv::Mat &src, cv::Mat &dst,
                       cv::ColorConversionCodes code) {
  impl_->Execute(src, dst, code);
}

}  // namespace hwcv