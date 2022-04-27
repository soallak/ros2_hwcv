#include "vitis_algorithm.hpp"

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

  cl_int err;
  OCL_CALL(context_ = cl::Context(device_, NULL, NULL, NULL, &err), err);
  OCL_CALL(queue_ = cl::CommandQueue(context_, device_,
                                     CL_QUEUE_PROFILING_ENABLE, &err),
           err);

  auto binaries = xcl::import_binary_file(GetBinaryFile());
  OCL_CALL(program_ = cl::Program(context_, devices, binaries, NULL, &err),
           err);
  is_binary_loaded_ = true;
}

}  // namespace hwcv