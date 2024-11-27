#ifndef TOY_NN_TENSOR_TENSOR_CREATOR_H__
#define TOY_NN_TENSOR_TENSOR_CREATOR_H__
#include "tensor.h"
#include <vector>

namespace toytorch {

Tensor empty(const TensorShape &shape, bool requires_grad = false);
Tensor zero(const TensorShape &shape, bool requires_grad = false);
Tensor ones(const TensorShape &shape, bool requires_grad = false);
Tensor arange(float start, float end, float step = 1, bool requires_grad = false);

// Uniform distribution
Tensor rand(const TensorShape &shape, bool requires_grad = false);

// Normal distribution
Tensor randn(const TensorShape &shape, bool requires_grad = false);

Tensor empty_like(const Tensor &input, bool requires_grad = false);
Tensor zero_like(const Tensor &input, bool requires_grad = false);
Tensor ones_like(const Tensor &input, bool requires_grad = false);
Tensor rand_like(const Tensor &input, bool requires_grad = false);
Tensor randn_like(const Tensor &input, bool requires_grad = false);

} // namespace toytorch

#endif // TOY_NN_TENSOR_TENSOR_CREATOR_H__