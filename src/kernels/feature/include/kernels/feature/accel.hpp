#pragma once
#include <ap_int.h>

#include <string>

#include "feature_config.hpp"

constexpr auto XF_PTR_SRC_WIDTH = 128;
constexpr auto XF_PTR_DST_WIDTH = 128;

extern "C" {

void fast_accel(ap_uint<XF_PTR_SRC_WIDTH>* src, ap_uint<XF_PTR_DST_WIDTH>* dst,
                int height, int width, int threshold);
}

namespace hwcv::kernels::feature {

namespace fast {

constexpr size_t GetMaxHeight() { return XF_HEIGHT; }
constexpr size_t GetMaxWidth() { return XF_WIDTH; }
constexpr bool GetNMS() { return XF_FAST_NMS; }
std::string GetName();

}  // namespace fast

std::string GetBinaryFile();

}  // namespace hwcv::kernels::feature