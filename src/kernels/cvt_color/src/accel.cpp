#include <kernels/cvt_color/accel.hpp>

#include "cvt_color_config.hpp"

namespace hwcv::kernels::cvt_color {

size_t bgr2gray::GetMaxHeight() { return XF_HEIGHT; }
size_t bgr2gray::GetMaxWidth() { return XF_WIDTH; }
std::string bgr2gray::GetName() {
#if defined(CVT_COLOR_BGR2GRAY_NAME)
  return CVT_COLOR_BGR2GRAY_NAME;
#else
  return "";
#endif
}

size_t demosaicing::GetMaxHeight() { return XF_HEIGHT; }
size_t demosaicing::GetMaxWidth() { return XF_WIDTH; }
std::string demosaicing::GetName() {
#if defined(CVT_COLOR_DEMOSAICING_NAME)
  return CVT_COLOR_DEMOSAICING_NAME;
#else
  return "";
#endif
}

std::string GetBinaryFile() {
#if defined(CVT_COLOR_BINARY_FILE)
  return CVT_COLOR_BINARY_FILE;
#else
  return "";
#endif
}

}  // namespace hwcv::kernels::cvt_color