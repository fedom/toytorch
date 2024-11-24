#ifndef TOY_NN_TENSOR_TENSOR_CREATOR_H__
#define TOY_NN_TENSOR_TENSOR_CREATOR_H__
#include "tensor.h"
#include <vector>

namespace toytorch {

Tensor empty(const std::vector<int> &shape, bool requires_grad = false);
Tensor zero(const std::vector<int> &shape, bool requires_grad = false);
Tensor ones(const std::vector<int> &shape, bool requires_grad = false);

// Uniform distribution
Tensor rand(const std::vector<int> &shape, bool requires_grad = false);

// Normal distribution
Tensor randn(const std::vector<int> &shape, bool requires_grad = false);

Tensor empty_like(const Tensor &input, bool requires_grad = false);
Tensor zero_like(const Tensor &input, bool requires_grad = false);
Tensor ones_like(const Tensor &input, bool requires_grad = false);
Tensor rand_like(const Tensor &input, bool requires_grad = false);
Tensor randn_like(const Tensor &input, bool requires_grad = false);

} // namespace toytorch

#endif // TOY_NN_TENSOR_TENSOR_CREATOR_H__