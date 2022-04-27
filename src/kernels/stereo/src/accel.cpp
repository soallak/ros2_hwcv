#include <kernels/stereo/accel.hpp>

#include "stereo_lbm_config.hpp"

namespace hwcv::kernels::stereo {

std::string lbm::GetName() {
#if defined(STEREO_LBM_NAME)
  return STEREO_LBM_NAME;
#else
  return "";
#endif
}

std::string GetBinaryFile() {
#if defined(STEREO_BINARY_FILE)
  return STEREO_BINARY_FILE;
#else
  return "";
#endif
}

}  // namespace hwcv::kernels::stereo