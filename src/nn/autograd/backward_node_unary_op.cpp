#include "nn/autograd/backward_node_unary_op.h"
#include "nn/exceptions/exceptions.h"
#include "nn/operations/tensor_operations.h"

namespace toytorch::autograd {

Tensor ExpBackward::calculate_grad(Tensor grad, Tensor input) {
  return mul(grad, input);
}

Tensor LogBackward::calculate_grad(Tensor grad, Tensor input) {
  return div(grad, input);
}

Tensor NegBackward::calculate_grad(Tensor grad, Tensor input) {
  return neg(grad);
}

Tensor AbsBackward::calculate_grad(Tensor grad, Tensor input) {
  return grad * sign(input);
}

Tensor BernoulliBackward::calculate_grad(Tensor grad, Tensor input) {
  // Straight-Through Estimator (STE)
  return grad;
}

Tensor SumBackward::calculate_grad(Tensor grad, Tensor input) {
  return mul(grad, Tensor(input.shape(), 1));
}

Tensor ViewBackward::calculate_grad(Tensor grad, Tensor input) {
  return grad.view(input.shape());
}

Tensor Pad2dBackward::calculate_grad(Tensor grad, Tensor input) {
  int height_dim = grad.dim() - 2;
  int width_dim = grad.dim() - 1;

  int height_slice_start = top_;
  int height_slice_end = grad.dim(height_dim) - bottom_;

  int width_slice_start = left_;
  int width_slice_end = grad.dim(width_dim) - right_;

  return grad.slice(height_dim, height_slice_start, height_slice_end)
      .slice(width_dim, width_slice_start, width_slice_end);
}

Tensor Pad1dBackward::calculate_grad(Tensor grad, Tensor input) {

  int slice_start = left_;
  int slice_end = grad.dim(-1) - right_;

  return grad.slice(-1, slice_start, slice_end);
}

}  // namespace toytorch::autograd