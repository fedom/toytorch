#ifndef TOYTORCH_NN_TENSOR_CONVOLUTION_H__
#define TOYTORCH_NN_TENSOR_CONVOLUTION_H__
#include <cassert>
#include "nn/tensor/tensor.h"

namespace toytorch {

// https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
// https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
/**
 * @brief Calculate 2d convolution
 * 
 * @param input The input with shape [batch, in_channel, height, width]
 * @param weight The kernel with shape [out_channel, in_channel, kernel_height, kernel_width]
 * @param stride
 * @param padding
 * @return Tensor 
 */
Tensor conv2d(const Tensor& input, const Tensor& weight,
              const std::array<int, 2>& stride = {1},
              const std::array<int, 2>& padding = {0});

}  // namespace toytorch

#endif  // TOYTORCH_NN_TENSOR_CONVOLUTION_H__