
#include "nn/autograd/backward_node_dropout_op.h"
#include "nn/exceptions/exceptions.h"
#include "nn/operations/tensor_operations.h"

namespace toytorch::autograd {

Tensor DropoutBackward::calculate_grad(Tensor grad, Tensor input) {
  return grad * mask_ * scale_;
}

Tensor Dropout2dBackward::calculate_grad(Tensor grad, Tensor input) {
  return grad * mask_;
}

} // namespace toytorch::autograd