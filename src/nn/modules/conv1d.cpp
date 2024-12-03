#include "nn/modules/conv1d.h"
#include <iostream>
#include "nn/operations/convolution.h"
#include "nn/operations/tensor_operations.h"
#include "nn/tensor/tensor_creator.h"

namespace toytorch::nn {

Conv1d::Conv1d(int in_channels, int out_channels, int kernel_size, int stride,
               const std::array<int, 2>& padding, bool bias)
    : padding_(padding), stride_(stride) {
  weights_ = randn({out_channels, in_channels, kernel_size}, true);

  if (bias) {
    bias_ = randn({out_channels, 1}, true);
  }
}

Tensor Conv1d::forward(const Tensor& input) const {

  Tensor result = conv1d(input, weights_, stride_, padding_);

  if (bias_.numel()) {
    result = result + bias_;
  }
  return result;
}

}  // namespace toytorch