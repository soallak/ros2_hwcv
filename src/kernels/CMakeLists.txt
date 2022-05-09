
project(hwcv_kernels)

# Make the finary file name known
if(DEFINED ROS_VITIS)
  find_package(ament_vitis REQUIRED)
  set(HWCV_KERNELS_BINARY_FILE
      "/lib/firmware/xilinx/${PROJECT_NAME}/${PROJECT_NAME}.xclbin")
  set(HWCV_KERNELS_CFG_DIR "src/kernels/cfg/")
  set(HWCV_KERNELS_CFG ${HWCV_KERNELS_CFG_DIR}/kv260.cfg)
  set(HWCV_KERNELS_DTSI ${HWCV_KERNELS_CFG_DIR}/kv260.dtsi)

endif()

# Each directory defines and synthesize its own kernels
add_subdirectory(cvt_color)
add_subdirectory(stereo)
add_subdirectory(feature)

# Link all kernels
if(NOT NOKERNEL AND DEFINED ROS_VITIS)
  set(HWCV_KERNELS ${HWCV_KERNERLS_CVT_COLOR} ${HWCV_KERNERLS_STEREO}
                   ${HWCV_KERNERLS_FEATURE})
  message(WARNING "Linking kernels ${HWCV_KERNELS}. This might take a while")
  vitis_link_kernel(
    OUTPUT
    ${PROJECT_NAME}
    KERNELS
    ${HWCV_KERNELS}
    CONFIG
    ${HWCV_KERNELS_CFG}
    DTSI
    ${HWCV_KERNELS_DTSI})

endif()
