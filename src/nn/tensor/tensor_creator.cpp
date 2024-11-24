#include "nn/tensor/tensor_creator.h"

namespace toytorch {

Tensor empty(const std::vector<int> &shape, bool requires_grad) {
  return Tensor(shape, 0, requires_grad);
}

Tensor empty_like(const Tensor &input, bool requires_grad) {
  return empty(input.shape(), requires_grad);
}

Tensor zero(const std::vector<int> &shape, bool requires_grad) {
  return Tensor(shape, 0, requires_grad);
}

Tensor zero_like(const Tensor &input, bool requires_grad) {
  return zero(input.shape(), requires_grad);
}

Tensor ones(const std::vector<int> &shape, bool requires_grad) {
  return Tensor(shape, 1, requires_grad);
}

Tensor ones_like(const Tensor &input, bool requires_grad) {
  return ones(input.shape(), requires_grad);
}

Tensor rand(const std::vector<int> &shape, bool requires_grad) {
  UniformRandomGenerator rg(0, 1);
  return Tensor(shape, rg, requires_grad);
}

Tensor randn(const std::vector<int> &shape, bool requires_grad) {
  NormalRandomGenerator rg(0, 1);
  return Tensor(shape, rg, requires_grad);
}

Tensor rand_like(const Tensor &input, bool requires_grad) {
  UniformRandomGenerator rg(0, 1);
  return Tensor(input.shape(), rg, requires_grad);
}

Tensor randn_like(const Tensor &input, bool requires_grad) {
  NormalRandomGenerator rg(0, 1);
  return Tensor(input.shape(), rg, requires_grad);
}

} // namespace toytorch