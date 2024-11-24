#ifndef SRC_NN_AUTOGRAD_AUTOGRAD_UTILS_H__
#define SRC_NN_AUTOGRAD_AUTOGRAD_UTILS_H__
#include "nn/tensor/tensor.h"
namespace toytorch::autograd {

Tensor shrink_broadcasted_grad(const Tensor& grad, const Tensor& target);

}  // namespace toytorch::autograd

#endif  // SRC_NN_AUTOGRAD_AUTOGRAD_UTILS_H__