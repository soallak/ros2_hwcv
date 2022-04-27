//#include <kernels/cvt_color/accel.hpp>
#include <vitis_common/common/xf_common.hpp>
#include <vitis_common/common/xf_utility.hpp>
#include <vitis_common/imgproc/xf_cvt_color.hpp>

#include "cvt_color_config.hpp"

namespace {

constexpr auto XF_MAT_SRC_TYPE = XF_8UC3;
constexpr auto XF_MAT_DST_TYPE = XF_8UC1;
constexpr auto XF_PTR_SRC_WIDTH = 128;
constexpr auto XF_PTR_DST_WIDTH = 128;

}  // namespace

extern "C" {

void bgr2gray_accel(ap_uint<XF_PTR_SRC_WIDTH>* src,
                    ap_uint<XF_PTR_DST_WIDTH>* dst, int height, int width) {
#pragma HLS INTERFACE m_axi port = src offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = dst offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = height
#pragma HLS INTERFACE s_axilite port = width
#pragma HLS INTERFACE s_axilite port = return

  xf::cv::Mat<XF_MAT_SRC_TYPE, XF_HEIGHT, XF_WIDTH, XF_NPPC1> src_mat(height,
                                                                      width);
  xf::cv::Mat<XF_MAT_DST_TYPE, XF_HEIGHT, XF_WIDTH, XF_NPPC1> dst_mat(height,
                                                                      width);

#pragma HLS STREAM variable = src_mat.data depth = 2
#pragma HLS STREAM variable = dst_mat.data depth = 2
#pragma HLS DATAFLOW

  xf::cv::Array2xfMat<XF_PTR_SRC_WIDTH, XF_MAT_SRC_TYPE, XF_HEIGHT, XF_WIDTH,
                      XF_NPPC1>(src, src_mat);

  xf::cv::bgr2gray<XF_MAT_SRC_TYPE, XF_MAT_DST_TYPE, XF_HEIGHT, XF_WIDTH,
                   XF_NPPC1>(src_mat, dst_mat);

  xf::cv::xfMat2Array<XF_PTR_DST_WIDTH, XF_MAT_DST_TYPE, XF_HEIGHT, XF_WIDTH,
                      XF_NPPC1>(dst_mat, dst);
  return;
}
}