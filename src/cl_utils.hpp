#pragma once

#include <spdlog/fmt/fmt.h>

#include <opencv2/core.hpp>
#include <vitis_common/xcl2/xcl2.hpp>

#include "except.hpp"

#define OCL_CALL(call, err)                                                  \
  call;                                                                      \
  if (err != CL_SUCCESS) {                                                   \
    throw CLError(fmt::format(FMT_STRING("Call {} failed with error code "), \
                              #call, err));                                  \
  }

namespace hwcv {

size_t GetBufferSizeForCvMat(cv::Mat const& mat);

cl::Buffer CreateBufferForCvMat(cv::Mat const& mat, cl::Context context,
                                cl_mem_flags mem_flags);

template <typename T>
void SetKernelArg(cl::Kernel kernel, cl_uint idx, T arg) {
  cl_uint err = 0;
  OCL_CALL(err = kernel.setArg(idx, arg), err);
}

template <typename T, typename... Ts>
void SetKernelArg(cl::Kernel kernel, cl_uint start_idx, T arg, Ts... args) {
  SetKernelArg(kernel, start_idx, arg);
  SetKernelArg(kernel, start_idx + 1, args...);
}

}  // namespace hwcv