#include "cl_utils.hpp"

namespace hwcv {
size_t GetBufferSizeForCvMat(cv::Mat const& mat) {
  const auto bytes = mat.rows * mat.cols * mat.elemSize();
  return bytes;
}
cl::Buffer CreateBufferForCvMat(cv::Mat const& mat, cl::Context context,
                                cl_mem_flags mem_flags) {
  cl_int err = 0;
  auto bytes = GetBufferSizeForCvMat(mat);
  OCL_CALL(cl::Buffer buffer(context, mem_flags, bytes, NULL, &err), err);
  return buffer;
}
}  // namespace hwcv