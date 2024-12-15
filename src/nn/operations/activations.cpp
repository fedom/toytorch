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


Tensor softmax(const Tensor& input, int dim) {
  dim = normalize_dim(input, dim);
  if (dim < 0 || dim >= input.dim()) {
    throw ExceptionInvalidArgument("log_softmax dim out of range");
  }

  Tensor output = [&]() {
    autograd::GradModeGuard nograd;
    Tensor input_exp = exp(input);
    return input_exp / input_exp.sum(dim, true);
  }();

  UPDATE_BACKWARD_GRAPH_2(output, SoftmaxBackward, dim, output, input);

  return output;
}

Tensor log_softmax(const Tensor& input, int dim) {
  dim = normalize_dim(input, dim);
  if (dim < 0 || dim >= input.dim()) {
    throw ExceptionInvalidArgument("log_softmax dim out of range");
  }

  Tensor output = [&]() {
    autograd::GradModeGuard nograd;
    Tensor input_exp = exp(input);
    return input - log(input_exp.sum(dim, true));
  }();

  UPDATE_BACKWARD_GRAPH_1(output, LogSoftmaxBackward, dim, input);

  return output;
}
}

