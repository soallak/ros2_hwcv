## HWCV

A computer vision Vitis accelerated library 


## Accelerated Kernels

| Header    | Kernel            |
|-----------|-------------------|
| cvt_color | bgr2gray_accel    |
|           | demosaicing_accel |
| feature   | fast_accel        |
| stereo    | stereo_lbm_accel  |


## Disabling Kernel Synthesis 

It is possible to disable all kernel synthesis and linking by specify the cmake switch `ǸOKERNELS`.
When using `colcon` this can be done by appending `-DNOKERNELS=ON` to `--cmake-args` of `colcon
build`

Other switches are availble to selectively disable particular kernels:

| Switch              | Kernel            |
|---------------------|-------------------|
| NO_CVT_COLOR_KERNEL | bgr2gray_accel    |
|                     | demosaicing_accel |
| NO_FEATURE_KERNEL   | fast_accel        |
| NO_STEREO_KERNEL    | stereo_lbm_accel  |
