#include "vitis_algorithm.hpp"

#include <vitis_common/common/utilities.hpp>
#include <vitis_common/xcl2/xcl2.hpp>

#include "cl_utils.hpp"
#include "except.hpp"
#include "logger.hpp"

namespace hwcv {

VitisAlgorithm::VitisAlgorithm() : logger_(GetLogger()) {}

VitisAlgorithm::~VitisAlgorithm() { queue_.finish(); }

void VitisAlgorithm::LoadBinaryFile() {
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
  // TODO(soallak): is there a mem leak of file_buf ?
  char* file_buf = read_binary_file(GetBinaryFile(), file_buf_size);
  cl::Program::Binaries binaries{{file_buf, file_buf_size}};

  logger_->debug(
      fmt::format(FMT_STRING("Create program for binary {}"), GetBinaryFile()));
  OCL_CALL(program_ = cl::Program(context_, devices, binaries, NULL, &err),
           err);
  is_binary_loaded_ = true;
  logger_->debug("Finish program creating");
}

}  // namespace hwcv