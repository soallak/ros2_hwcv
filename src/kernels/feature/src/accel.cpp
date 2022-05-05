#include <kernels/feature/accel.hpp>

#include "feature_config.hpp"

namespace hwcv::kernels::feature {
std::string fast::GetName() {
#if defined(FEATURE_FAST_NAME)
  return FEATURE_FAST_NAME;
#else
  return "";
#endif
}

std::string GetBinaryFile() {
#if defined(FEATURE_BINARY_FILE)
  return FEATURE_BINARY_FILE;
#else
  return "";
#endif
}

}  // namespace hwcv::kernels::feature