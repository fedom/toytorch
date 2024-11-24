
#include "autograd_utils.h"

namespace toytorch::autograd {

Tensor shrink_broadcasted_grad(const Tensor& grad, const Tensor& target) {

  TensorShape grad_shape = grad.shape();
  TensorShape target_shape = target.shape();

  std::vector<int> dims;
  for (int i = 0; i < grad_shape.size(); i++) {
    if (target_shape[i] == 1 && grad_shape[i] != 1) {
      dims.push_back(i);
    }
  }

  return grad.sum(dims, true);
}

}  // namespace toytorch::autograd
