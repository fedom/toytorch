#include "nn/autograd/backward_node_activation_op.h"
#include "nn/tensor/tensor_creator.h"
#include "nn/operations/tensor_operations.h"
#include "nn/operations/activations.h"
#include <iostream>

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

Tensor SoftmaxBackward::calculate_grad(Tensor grad, Tensor input) {
  // assume the input is X = [x_1, x_2, ..., x_n]. 
  // Y = σ(X) = [y_1, y_2, ..., y_n].
  // So the Jacobian matrix is
  //   [[∂y_1/∂x_1, ∂y_1/∂x_2, ..., ∂y_1/∂x_n],
  //    [∂y_2/∂x_1, ∂y_2/∂x_2, ..., ∂y_2/∂x_n],
  //    [...]
  //    [∂y_n/∂x_1, ∂y_n/∂x_2, ..., ∂y_n/∂x_n]]
  // 
  // for i, j in ∂y_i/∂x_j, the partial derivative has two cases,
  //  (1) i = j, ∂y_i/∂x_j = ∂(e^x_i/Σe^x)/∂x_j = softmax(x_i)(1 - softmax(x_i)) = y_i * (1 - y_i)
  //  (2) i != j, ∂y_i/∂x_j = ∂(e^x_i/Σe^x)/∂x_j = -softmax(x_i)*softmax(x_j) = -(y_i * y_j)

  // So let calculate ∂L/∂x_i:
  // ∂L/∂x_1 = (∂L/∂y_1)*(∂y_1/∂x_1) + ... + (∂L/∂y_n)*(∂y_n/∂x_1)
  //        = grad_1 * y_1 * (1 - y_1) + ... + grad_n * (-y_n * y_1)
  //        = grad_1 * y_1 - grad_1 * y_1 * y_1 - ... - grad_n * y_n * y_1
  //        = grad_1 * y_1 - (grad_1 * y_1 + grad_2 * y_2 + ... + grad_n * y_n) * y_1
  //        = grad_1 * y_1 - sum(Grad * Y, dim_) * y_1
  // Note here, sum(Grad * Y, dim_) are the same for each element. Let's use C to represent it.
  // So that patial derivative for each elelment should be 
  //    ∂L/∂x_i = grad_i * y_i - C * y_i = (grad_i - C) * y_i
  // The formula for the whole tensor operator should be:
  //  ∂L/∂X = (Grad - C) * Y
  Tensor C = sum(grad * y_, dim_, true);
  Tensor result = (grad - C) * y_;

  return result;
}

Tensor LogSoftmaxBackward::calculate_grad(Tensor grad, Tensor input) {
  // y_i = logsoftmax(x_i) = x_i - log(Σe^x_j)
  // for i, j in ∂y_i/∂x_j, the partial derivative has two cases,
  // (1) i = j, ∂y_i/∂x_j = 1 - softmax(x_i)
  // (2) i != j, ∂y_i/∂x_j = -softmax(x_i)
  
  // So let calculate ∂L/∂xi:
  // Suppose ∂L/∂Y = [grad_1, grad_2, .. grad_n]
  // ∂L/∂x_1 = (∂L/∂y_1)*(∂y_1/∂x_1) + ... + (∂L/∂y_n)*(∂y_n/∂x_1)
  //        = grad_1 * (1 - softmax(x_1)) + grad_2 * (-softmax(x_1)) + ... + grad_n * (-softmax(x_1))
  //        = grad_1 - softmax(x_1) * (grad_1 + grad_2 + ... + grad_n)
  //        = grad_1 - softmax(x_1) * sum(Grad, dim_)
 
  // So the formula for the whole tensor operator should be:
  //    ∂L/∂X = Grad - softmax(X) * sum(Grad, dim_)

  return grad - sum(grad, dim_, true) * softmax(input, dim_);
}

} // namespace toytorch::autograd