#include "nn/modules/conv2d.h"
#include "nn/operations/convolution.h"
#include "nn/operations/tensor_operations.h"
#include "nn/tensor/tensor_creator.h"
#include <iostream>

namespace toytorch {

Conv2d::Conv2d(int in_channels, int out_channels,
               const std::array<int, 2>& kernel_size,
               const std::array<int, 2>& stride,
               const std::array<int, 4>& padding, bool bias)
    : padding_(padding), stride_(stride) {

  weights_ =
      randn({out_channels, in_channels, kernel_size[0], kernel_size[1]}, true);
  if (bias) {
    bias_ = randn({out_channels, 1, 1}, true);
  }
}

Tensor Conv2d::forward(const Tensor& input) const {

  Tensor result = conv2d(input, weights_, stride_, padding_);

  if (bias_.numel()) {
    result = result + bias_;
  }
  return result;
}

}  // namespace toytorch