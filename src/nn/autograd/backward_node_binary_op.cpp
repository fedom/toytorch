#include "nn/autograd/backward_node_binary_op.h"
#include "autograd_utils.h"
#include "nn/exceptions/exceptions.h"
#include "nn/tensor/tensor_creator.h"
#include "nn/operations/tensor_operations.h"
#include "nn/operations/matrix.h"

namespace toytorch::autograd {

// f(X) = X * W
// (1) ∂F(X)/∂X = W^T
//     ∂L/∂(X) = (∂L/∂f(X)) * (∂f(X)/∂X) = (∂L/∂f(X)) * W^T
// (2) ∂F(X)/∂W = X^T
//     ∂L/∂(W) = (∂L/∂f(X)) * (∂f(X)/∂X) = X^T * (∂L/∂f(X))
// Note here: While the chain rule does not matter in scalar case, because the multiplication
// for scalar is commutative. However, when it comes to matrix multiplication, the order does
// matter. The partial derivative for different operant is different.
Tensor MatmulBackward::calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) {
  return matmul(grad, rhs.transpose());
}
Tensor MatmulBackward::calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) {
  return matmul(lhs.transpose(), grad);
}

Tensor AddBackward::calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) {
  if (grad.shape() != lhs.shape()) {
    return shrink_broadcasted_grad(grad, lhs);
  }
  return grad;
}
Tensor AddBackward::calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) {
  if (grad.shape() != rhs.shape()) {
    return shrink_broadcasted_grad(grad, rhs);
  }
  return grad;
}

Tensor SubBackward::calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) {
  if (grad.shape() != lhs.shape()) {
    return shrink_broadcasted_grad(grad, lhs);
  }
  return grad;
}
Tensor SubBackward::calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) {
  if (grad.shape() != rhs.shape()) {
    return shrink_broadcasted_grad(grad, rhs);
  }
  return neg(grad);
}

Tensor MulBackward::calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) {
  Tensor result = mul(grad, rhs);

  if (result.shape() != lhs.shape()) {
    return shrink_broadcasted_grad(result, lhs);
  }
  return result;
}
Tensor MulBackward::calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) {
  Tensor result = mul(grad, lhs);

  if (result.shape() != rhs.shape()) {
    return shrink_broadcasted_grad(result, rhs);
  }
  return result;
}

Tensor DivBackward::calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) {
  Tensor result = div(grad, rhs);

  if (result.shape() != lhs.shape()) {
    return shrink_broadcasted_grad(result, lhs);
  }
  return result;
}
Tensor DivBackward::calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) {
  Tensor result = neg(mul(grad, lhs));
  result = div(result, mul(rhs, rhs));

  if (result.shape() != rhs.shape()) {
    return shrink_broadcasted_grad(result, rhs);
  }
  return result;
}

Tensor PowBackward::calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) {
  Tensor& base = lhs;
  Tensor& exp = rhs;

  Tensor new_exp = sub(exp, Tensor(1));
  Tensor result = pow(base, new_exp);
  result = mul(exp, result);
  result = mul(grad, result);

  // do we need this here?
  if (result.shape() != base.shape()) {
    result = shrink_broadcasted_grad(result, base);
  }
  return result;
}

Tensor PowBackward::calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) {
  // TODO(Leo): add implementation
  return Tensor();
}

Tensor WhereBackward::calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) {
  Tensor result = where(condition_, grad, zero_like(grad));

  if (result.shape() != lhs.shape()) {
    return shrink_broadcasted_grad(result, lhs);
  }
  return result;
}
Tensor WhereBackward::calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) {
  Tensor result = where(condition_, zero_like(grad), grad);

  if (result.shape() != rhs.shape()) {
    return shrink_broadcasted_grad(result, rhs);
  }
  return result;
}

}  // namespace toytorch::autograd