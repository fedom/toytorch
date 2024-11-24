#include "nn/autograd/node.h"
#include "nn/tensor/tensor.h"

namespace toytorch::autograd {

std::vector<Tensor> UnaryNode::apply(std::vector<Tensor>&& grads) {
  std::vector<Tensor> grads_result;

  Tensor& grad = grads[0];

  auto tensor = edges_[0].get_tensor();

  if (tensor.requires_grad()) {
    grads_result.push_back(calculate_grad(grad, tensor));
  } else {
    grads_result.push_back(Tensor());
  }

  return grads_result;
}

std::vector<Tensor> BinaryNode::apply(std::vector<Tensor>&& grads) {
  std::vector<Tensor> grads_result;
  Tensor& grad = grads[0];

  auto lhs = edges_[0].get_tensor();
  auto rhs = edges_[1].get_tensor();

  if (lhs.requires_grad()) {
    grads_result.push_back(calculate_lhs_grad(grad, lhs, rhs));
  } else {
    grads_result.push_back(Tensor());
  }

  if (rhs.requires_grad()) {
    grads_result.push_back(calculate_rhs_grad(grad, lhs, rhs));
  } else {
    grads_result.push_back(Tensor());
  }
  return grads_result;
}

}  // namespace toytorch::autograd