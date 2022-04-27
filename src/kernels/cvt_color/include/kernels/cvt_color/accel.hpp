#pragma once
#include <ap_int.h>

#include <string>
#include <vitis_common/common/xf_params.hpp>

constexpr auto XF_PTR_SRC_WIDTH = 128;
constexpr auto XF_PTR_DST_WIDTH = 128;

extern "C" {

/**
 * @brief Accelerated debayering function that uses Vitis Vision libraries
 *
 * @param src Input image data pointer
 * @param dst Output image data pointer
 * @param height Image height
 * @param width  Image width
 * @param pattern Input image bayer pattern
 */
void demosaicing_accel(ap_uint<XF_PTR_SRC_WIDTH>* src,
                       ap_uint<XF_PTR_DST_WIDTH>* dst, int height, int width,
                       XF_demosaicing pattern);

/**
 * @brief Accelerated BGR to Grayscale color conversion function that use Vitis
 * Vision libraries
 *
 * @param src Input image data pointer
 * @param dst Output image data pointer
 * @param height  Image height
 * @param width Image width
 */

void bgr2gray_accel(ap_uint<XF_PTR_SRC_WIDTH>* src,
                    ap_uint<XF_PTR_DST_WIDTH>* dst, int height, int width);
}

namespace hwcv::kernels::cvt_color {

namespace demosaicing {

size_t GetMaxHeight();
size_t GetMaxWidth();
std::string GetName();

}  // namespace demosaicing

namespace bgr2gray {

size_t GetMaxHeight();
size_t GetMaxWidth();
std::string GetName();

}  // namespace bgr2gray

std::string GetBinaryFile();

// namespace bgr2gray

}  // namespace hwcv::kernels::cvt_color