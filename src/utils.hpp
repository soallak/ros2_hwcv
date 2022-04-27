#pragma once
#include <opencv2/core.hpp>
#include <type_traits>
#include <vitis_common/common/xf_common.hpp>
#include <vitis_common/common/xf_utility.hpp>

namespace hwcv {
/**
 * @brief
 *
 * @tparam TYPE xf::cv::Mat type
 * @tparam ROWS Maximum Rows
 * @tparam COLS Maximum Cols
 * @tparam NPPC xf::cv::Mat number of pixels per channel
 * @param mat input opencv mat to wrap
 * @note mat is not a const reference to indicate that the wrapper result can be
 * used for writing.
 * @return xf::cv::Mat<TYPE, ROWS, COLS, NPPC> wrapper xf matrice
 */
template <int TYPE, int ROWS, int COLS, int NPPC>
xf::cv::Mat<TYPE, ROWS, COLS, NPPC> CvMat2XfMat(cv::Mat const& mat) {
  xf::cv::Mat<TYPE, ROWS, COLS, NPPC> xf_mat(mat.rows, mat.cols);

  xf_mat.copyTo(static_cast<void*>(mat.data));
  return xf_mat;
}
/**
 * @brief Allocates data buffer based to hold all data of an xf::Mat
 *
 * @tparam PTR_WIDTH Witdth of the ap_uint
 * @tparam TYPE Type of xf::cv::Mat
 * @tparam ROWS Maximum Rows
 * @tparam COLS Maximum Cols
 * @tparam NPPC Number of pixels per channel
 * @param mat matrice to create buffer for and copy from
 * @return ap_uint<PTR_WIDTH>* pointer to the allocated data
 */
template <int PTR_WIDTH, int TYPE, int ROWS, int COLS, int NPPC>
ap_uint<PTR_WIDTH>* CreateApUintData(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& mat) {
  constexpr auto pixel_width = XF_PIXELWIDTH(TYPE, NPPC);

  // TODO(soallak) Allocate only what is needed using mat.rows and mat.cols
  constexpr auto buffer_size =
      (ROWS * COLS * pixel_width / PTR_WIDTH) +
      ((ROWS * COLS * pixel_width % PTR_WIDTH) ? 1 : 0);

  using mat_type = xf::cv::Mat<TYPE, ROWS, COLS, NPPC>;
  ap_uint<PTR_WIDTH>* ptr = new ap_uint<PTR_WIDTH>[buffer_size];

  xf::cv::xfMat2Array<PTR_WIDTH, TYPE, ROWS, COLS, NPPC>(mat, ptr);

  return ptr;
}

/**
 * @brief Allocates data buffer based to hold all data of an xf::Mat
 *
 * @tparam PTR_WIDTH Witdth of the ap_uint
 * @tparam TYPE Type of xf::cv::Mat
 * @tparam ROWS Maximum Rows
 * @tparam COLS Maximum Cols
 * @tparam NPPC Number of pixels per channel
 * @return ap_uint<PTR_WIDTH>* pointer to the allocated data
 */
template <int PTR_WIDTH, int TYPE, int ROWS, int COLS, int NPPC>
ap_uint<PTR_WIDTH>* CreateApUintData() {
  constexpr auto pixel_width = XF_PIXELWIDTH(TYPE, NPPC);

  constexpr auto buffer_size =
      (ROWS * COLS * pixel_width / PTR_WIDTH) +
      ((ROWS * COLS * pixel_width % PTR_WIDTH) ? 1 : 0);

  using mat_type = xf::cv::Mat<TYPE, ROWS, COLS, NPPC>;
  ap_uint<PTR_WIDTH>* ptr = new ap_uint<PTR_WIDTH>[buffer_size];

  return ptr;
}

/**
 * @brief Destroy allocated data buffer
 *
 * @tparam PTR_WIDTH Word width of data buffer elements
 * @param ptr ptr to data buffer
 */
template <int PTR_WIDTH>
void DestroyApUint(ap_uint<PTR_WIDTH>* ptr) {
  delete[] ptr;
}

template <typename T>
class InRange {
 public:
  template <T min, T max>
  static constexpr bool In(const T val) {
    return val >= min && val <= max;
  }
};

}  // namespace hwcv