#include "nn/autograd/backward_node_activation_op.h"
#include "nn/tensor/tensor_operations.h"
#include "nn/tensor/tensor_creator.h"
#include "nn/tensor/tensor_operations.h"

namespace toytorch::autograd {

Tensor SigmoidBackward::calculate_grad(Tensor grad, Tensor input) {
  // f'(x) = f(x) * (1 - f(x)) when f(x) = sigmoid(x)
  Tensor t1 = sigmoid(input);
  Tensor t2 = sub(Tensor(1), sigmoid(input));

  Tensor result = mul(mul(t1, t2), grad);

  return result;
}

Tensor ReluBackward::calculate_grad(Tensor grad, Tensor input) {
  Tensor result = where(input >= Tensor(0), ones_like(input), zero_like(input));
  result = mul(grad, result);

  return result;
}

} // namespace toytorch::autograd