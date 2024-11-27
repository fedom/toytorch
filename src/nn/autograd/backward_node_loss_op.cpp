
#include "nn/autograd/backward_node_loss_op.h"
#include "nn/operations/tensor_operations.h"
#include "nn/tensor/tensor_creator.h"
#include "nn/operations/common_types.h"
#include "nn/exceptions/exceptions.h"

namespace toytorch::autograd {

// https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
Tensor SmoothL1LossBackward::calculate_lhs_grad(Tensor grad, Tensor lhs,
                                                Tensor rhs) {

  if (rt_ == ReductionType::Sum) {
    assert(grad.is_scalar());
    grad = grad * ones(lhs.shape());
  } else if (rt_ == ReductionType::Mean) {
    assert(grad.is_scalar());
    grad = grad * ones(lhs.shape()) / lhs.data_size();
  } else if (rt_ == ReductionType::None) {
    assert(grad.shape() == lhs.shape());
  } else {
    throw ExceptionInvalidArgument("Unrecognized ReductionType");
  }

  Tensor diff = lhs - rhs;

  // |lhs - rhs| < beta
  Tensor case1 = (diff / beta_) * grad;

  // |lhs - rhs| >= beta
  Tensor case2 = sign(diff) * grad;

  Tensor result = where(abs(diff) < beta_, case1, case2);

  return result;
}

Tensor SmoothL1LossBackward::calculate_rhs_grad(Tensor grad, Tensor lhs,
                                                Tensor rhs) {
  if (rt_ == ReductionType::Sum) {
    assert(grad.is_scalar());
    grad = grad * ones(lhs.shape());
  } else if (rt_ == ReductionType::Mean) {
    assert(grad.is_scalar());
    grad = grad * ones(lhs.shape()) / lhs.data_size();
  } else if (rt_ == ReductionType::None) {
    assert(grad.shape() == lhs.shape());
  } else {
    throw ExceptionInvalidArgument("Unrecognized ReductionType");
  }

  Tensor diff = lhs - rhs;

  // |lhs - rhs| < beta
  Tensor case1 = (neg(diff) / beta_) * grad;

  // |lhs - rhs| >= beta
  Tensor case2 = (neg(sign(diff)))*grad;

  Tensor result = where(abs(diff) < beta_, case1, case2);

  return result;
}

}  // namespace toytorch::autograd