#include "nn/autograd/backward_node_unary_op.h"
#include "nn/tensor/tensor_operations.h"

namespace toytorch::autograd {

Tensor ExpBackward::calculate_grad(Tensor grad, Tensor input) {
  return mul(grad, input);
}

Tensor NegBackward::calculate_grad(Tensor grad, Tensor input) {
  // return ll::mul(grad, Tensor(input.shape(), 1));
  return neg(grad);
}

Tensor AbsBackward::calculate_grad(Tensor grad, Tensor input) {
  // return ll::mul(grad, Tensor(input.shape(), 1));
  // TODO(Leo): add implementation
  return Tensor();
}

Tensor SumBackward::calculate_grad(Tensor grad, Tensor input) {
  return mul(grad, Tensor(input.shape(), 1));
}

Tensor ViewBackward::calculate_grad(Tensor grad, Tensor input) {
  return grad.view(input.shape());
}

}  // namespace toytorch::autograd