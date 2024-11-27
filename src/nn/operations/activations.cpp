#include "nn/operations/activations.h"
#include "nn/operations/tensor_operations.h"
#include "nn/autograd/autograd.h"
#include "nn/autograd/backward_node_activation_op.h"

namespace toytorch {

Tensor sigmoid(const Tensor& tensor) {
  // We want the sigmoid's backward node to be single op node in the graph rather
  // than a series of arithmetic nodes. So we disable the grad mode before we do
  // arithmetic operations and add a single SigmoidBackward node at last.
  Tensor result = [&]() {
    autograd::GradModeGuard grad_guard(false);
    return div(Tensor(1), add(Tensor(1), exp(neg(tensor))));
  }();

  UPDATE_BACKWARD_GRAPH(result, SigmoidBackward, tensor);

  return result;
}

Tensor relu(const Tensor& tensor) {

  // We want the relu's backward node to be a single op node in the graph rather
  // than a series of arithmetic nodes. So we disable the grad mode before we do
  // arithmetic operations and add a single ReluBackward node at last.

  // Tensor result = (input + abs(input)) / 2;
  // return result;
  Tensor result = [&]() {
    autograd::GradModeGuard grad_guard(false);
    return div(add(tensor, abs(tensor)), Tensor(2));
  }();

  UPDATE_BACKWARD_GRAPH(result, ReluBackward, tensor);

  return result;
}
}

