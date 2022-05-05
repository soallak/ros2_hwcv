#include "vitis_algorithm.hpp"

#include <boost/filesystem.hpp>
#include <vitis_common/common/utilities.hpp>
#include <vitis_common/xcl2/xcl2.hpp>

#include "cl_utils.hpp"
#include "except.hpp"
#include "logger.hpp"

namespace hwcv {

VitisAlgorithm::VitisAlgorithm() : logger_(GetLogger()) {}

VitisAlgorithm::~VitisAlgorithm() {
  queue_.finish();
  // TODO(soallak): delete file buffer
  // if (binary_file_buf_) delete[] binary_file_buf_;
}

void VitisAlgorithm::LoadBinaryFile() {
  namespace fs = boost::filesystem;
  if (!fs::is_regular_file(GetBinaryFile())) {
    throw CLError(
        fmt::format("Could not find appropriate binary {}", GetBinaryFile()));
  }

  auto devices = xcl::get_xil_devices();
  if (devices.empty()) {
    throw CLError("No XIL devices were found");
  } else {
    device_ = devices[0];
    devices.resize(1);
  }

  std::string device_name = device_.getInfo<CL_DEVICE_NAME>();
  logger_->debug(fmt::format(FMT_STRING("Found device {}"), device_name));

  cl_int err(0);
  OCL_CALL(context_ = cl::Context(device_, NULL, NULL, NULL, &err), err);
  OCL_CALL(queue_ = cl::CommandQueue(context_, device_,
                                     CL_QUEUE_PROFILING_ENABLE, &err),
           err);

  logger_->debug(
      fmt::format(FMT_STRING("Import binary file {}"), GetBinaryFile()));
  unsigned int file_buf_size = 0;
  binary_file_buf_ = read_binary_file(GetBinaryFile(), file_buf_size);
  cl::Program::Binaries binaries{{binary_file_buf_, file_buf_size}};

  logger_->debug(
      fmt::format(FMT_STRING("Create program for binary {}"), GetBinaryFile()));
  OCL_CALL(program_ = cl::Program(context_, devices, binaries, NULL, &err),
           err);
  is_binary_loaded_ = true;
  logger_->debug("Finish program creation");
}

}  // namespace hwcv