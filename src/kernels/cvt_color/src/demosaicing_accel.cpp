
//#include <kernels/cvt_color/accel.hpp>
#include <vitis_common/common/xf_common.hpp>
#include <vitis_common/common/xf_utility.hpp>
#include <vitis_common/imgproc/xf_demosaicing.hpp>

#include "cvt_color_config.hpp"

namespace {

constexpr auto XF_MAT_SRC_TYPE = XF_8UC1;
constexpr auto XF_MAT_DST_TYPE = XF_8UC4;
constexpr auto XF_PTR_SRC_WIDTH = 128;
constexpr auto XF_PTR_DST_WIDTH = 128;

}  // namespace

extern "C" {

void demosaicing_accel(ap_uint<XF_PTR_SRC_WIDTH>* src,
                       ap_uint<XF_PTR_DST_WIDTH>* dst, int height, int width,
                       XF_demosaicing pattern) {
#pragma HLS INTERFACE m_axi port = src offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = dst offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = height
#pragma HLS INTERFACE s_axilite port = width
#pragma HLS INTERFACE s_axilite port = pattern
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

  // TODO(soallak): Figure out why this hangs on HW
  // if (pattern == XF_BAYER_BG) {
  //   xf::cv::demosaicing<XF_BAYER_BG, XF_MAT_SRC_TYPE, XF_MAT_DST_TYPE,
  //                       XF_HEIGHT, XF_WIDTH, XF_NPPC1, XF_USE_URAM>(src_mat,
  //                                                                   dst_mat);
  // } else if (pattern == XF_BAYER_GB) {
  //   xf::cv::demosaicing<XF_BAYER_GB, XF_MAT_SRC_TYPE, XF_MAT_DST_TYPE,
  //                       XF_HEIGHT, XF_WIDTH, XF_NPPC1, XF_USE_URAM>(src_mat,
  //                                                                   dst_mat);
  // } else if (pattern == XF_BAYER_GR) {
  //   xf::cv::demosaicing<XF_BAYER_GR, XF_MAT_SRC_TYPE, XF_MAT_DST_TYPE,
  //                       XF_HEIGHT, XF_WIDTH, XF_NPPC1, XF_USE_URAM>(src_mat,
  //                                                                   dst_mat);
  // } else if (pattern == XF_BAYER_RG) {
  //   xf::cv::demosaicing<XF_BAYER_RG, XF_MAT_SRC_TYPE, XF_MAT_DST_TYPE,
  //                       XF_HEIGHT, XF_WIDTH, XF_NPPC1, XF_USE_URAM>(src_mat,
  //                                                                   dst_mat);
  // }
  xf::cv::demosaicing<XF_BAYER_BG, XF_MAT_SRC_TYPE, XF_MAT_DST_TYPE, XF_HEIGHT,
                      XF_WIDTH, XF_NPPC1, XF_USE_URAM>(src_mat, dst_mat);

  xf::cv::xfMat2Array<XF_PTR_DST_WIDTH, XF_MAT_DST_TYPE, XF_HEIGHT, XF_WIDTH,
                      XF_NPPC1>(dst_mat, dst);

  return;
}

}  // End of extern C