#include "nn/operations/dropout.h"
#include "nn/operations/tensor_operations.h"
#include "nn/autograd/autograd.h"
#include "nn/autograd/backward_node_dropout_op.h"
#include "nn/tensor/tensor_creator.h"
#include <iostream>

namespace toytorch {

Tensor dropout(const Tensor& input, float p, bool train) {

  float scale = 1;

  auto&& [output_r, mask_r] = [&]() {
    autograd::GradModeGuard grad_guard(false);

    Tensor mask;
    Tensor output;

    if (!train) {
      output = input.deep_copy();
      mask = ones_like(input);
    } else {
      float p1m = 1. - p;
      scale = (p1m == 0 ? 0 : 1. / p1m);

      mask = empty_like(input);
      mask.bernoulli_(p1m);
      output = input.mul(mask).mul_(scale);
    }
    return std::make_tuple(output, mask);
  }();

  UPDATE_BACKWARD_GRAPH_3(output_r, DropoutBackward, scale, train, mask_r,
                          input);

  return output_r;
}

Tensor dropout2d(const Tensor& input, float p, bool train) {

  // (N, C, H, W) or (N, C, L)
  if (input.dim() != 3 && input.dim() != 4) {
    throw ExceptionInvalidArgument(
        "dropout2d() input's dim can't be less than 3");
  }

  TensorShape mask_shape;
  mask_shape.push_back(input.dim(0));
  mask_shape.push_back(input.dim(1));

  Tensor mask = ones(mask_shape);

  mask = [&]() {
    autograd::GradModeGuard grad_guard(false);
    return dropout(mask, p, train);
  }();

  // for input.dim() == 3
  mask.unsqueeze_(-1);
  if (input.dim() == 4) {
    mask.unsqueeze_(-1);
  }

  std::cout << "1111\n";
  input.print_shape();
  mask.print_shape();
  // We don't need multiply scale here. Since it is already multiplied in dropout() and
  // reflected in mask.
  Tensor result = input * mask;

  UPDATE_BACKWARD_GRAPH_2(result, Dropout2dBackward, train, mask, input);

  return result;
}

}  // namespace toytorch