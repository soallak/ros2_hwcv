//#include <kernels/stereo/accel.hpp>
#include <vitis_common/imgproc/xf_stereolbm.hpp>

#include "stereo_lbm_config.hpp"

constexpr auto XF_MAT_SRC_TYPE = XF_8UC1;
constexpr auto XF_MAT_DST_TYPE = XF_16UC1;
constexpr auto XF_PTR_SRC_WIDTH = 128;
constexpr auto XF_PTR_DST_WIDTH = 128;

extern "C" {
void stereo_lbm_accel(ap_uint<XF_PTR_SRC_WIDTH>* left_src,
                      ap_uint<XF_PTR_SRC_WIDTH>* right_src,
                      ap_uint<XF_PTR_DST_WIDTH>* dst, int height, int width,
                      int pre_filter_cap, int min_disparity,
                      int uniqueness_ratio, int texture_threshold) {
#pragma HLS INTERFACE m_axi port = left_src offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = right_src offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = dst offset = slave bundle = gmem3
#pragma HLS INTERFACE s_axilite port = height
#pragma HLS INTERFACE s_axilite port = width
#pragma HLS INTERFACE s_axilite port = pre_filter_cap
#pragma HLS INTERFACE s_axilite port = min_disparity
#pragma HLS INTERFACE s_axilite port = uniqueness_ratio
#pragma HLS INTERFACE s_axilite port = texture_threshold
#pragma HLS INTERFACE s_axilite port = return

  xf::cv::Mat<XF_MAT_SRC_TYPE, XF_HEIGHT, XF_WIDTH, XF_NPPC1> left_src_mat(
      height, width);
  xf::cv::Mat<XF_MAT_SRC_TYPE, XF_HEIGHT, XF_WIDTH, XF_NPPC1> right_src_mat(
      height, width);
  xf::cv::Mat<XF_MAT_DST_TYPE, XF_HEIGHT, XF_WIDTH, XF_NPPC1> dst_mat(height,
                                                                      width);
  xf::cv::xFSBMState<XF_STEREO_LBM_WSIZE, XF_STEREO_LBM_NDISP,
                     XF_STEREO_LBM_NDISP_UNIT>
      bm_state;

  bm_state.preFilterCap = pre_filter_cap;
  bm_state.uniquenessRatio = uniqueness_ratio;
  bm_state.textureThreshold = texture_threshold;
  bm_state.minDisparity = min_disparity;

#pragma HLS STREAM variable = left_src_mat.data depth = 2
#pragma HLS STREAM variable = right_src_mat.data depth = 2
#pragma HLS STREAM variable = dst_mat.data depth = 2

#pragma HLS DATAFLOW

  xf::cv::Array2xfMat<XF_PTR_SRC_WIDTH, XF_MAT_SRC_TYPE, XF_HEIGHT, XF_WIDTH,
                      XF_NPPC1>(left_src, left_src_mat);
  xf::cv::Array2xfMat<XF_PTR_SRC_WIDTH, XF_MAT_SRC_TYPE, XF_HEIGHT, XF_WIDTH,
                      XF_NPPC1>(right_src, right_src_mat);

  xf::cv::StereoBM<XF_STEREO_LBM_WSIZE, XF_STEREO_LBM_NDISP,
                   XF_STEREO_LBM_NDISP_UNIT, XF_MAT_SRC_TYPE, XF_MAT_DST_TYPE,
                   XF_HEIGHT, XF_WIDTH, XF_NPPC1, XF_USE_URAM>(
      left_src_mat, right_src_mat, dst_mat, bm_state);

  xf::cv::xfMat2Array<XF_PTR_DST_WIDTH, XF_MAT_DST_TYPE, XF_HEIGHT, XF_WIDTH,
                      XF_NPPC1>(dst_mat, dst);

  return;
}
}