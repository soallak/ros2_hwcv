#pragma once
#include <ap_int.h>

#include <string>

#include "stereo_lbm_config.hpp"

constexpr auto XF_PTR_SRC_WIDTH = 128;
constexpr auto XF_PTR_DST_WIDTH = 128;

extern "C" {

void stereo_lbm_accel(ap_uint<XF_PTR_SRC_WIDTH>* left_src,
                      ap_uint<XF_PTR_SRC_WIDTH>* right_src,
                      ap_uint<XF_PTR_DST_WIDTH>* dst, int height, int width,
                      int pre_filter_cap, int min_disparity,
                      int uniqueness_ratio, int texture_threshold);
}

namespace hwcv::kernels::stereo {

namespace lbm {

constexpr size_t GetMaxHeight() { return XF_HEIGHT; }
constexpr size_t GetMaxWidth() { return XF_WIDTH; }
constexpr size_t GetMaxDisparity() { return XF_WIDTH - XF_STEREO_LBM_NDISP; }
std::string GetName();

}  // namespace lbm

std::string GetBinaryFile();

}  // namespace hwcv::kernels::stereo