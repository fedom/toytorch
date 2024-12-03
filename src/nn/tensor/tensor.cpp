#include "nn/tensor/tensor.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include "nn/autograd/autograd.h"
#include "nn/autograd/backward_node_unary_op.h"
#include "nn/exceptions/exceptions.h"
#include "nn/operations/tensor_helper.h"
#include "nn/operations/tensor_operations.h"

namespace toytorch {

using autograd::GradInfo;

Tensor Tensor::add(const Tensor& other) const {
  return toytorch::add(*this, other);
}

Tensor Tensor::sub(const Tensor& other) const {
  return toytorch::sub(*this, other);
}

Tensor Tensor::mul(const Tensor& other) const {
  return toytorch::mul(*this, other);
}

Tensor Tensor::div(const Tensor& other) const {
  return toytorch::div(*this, other);
}

Tensor Tensor::squeeze(int dim) const {
  return toytorch::squeeze(*this, dim);
}
Tensor Tensor::unsqueeze(int dim) const {
  return toytorch::unsqueeze(*this, dim);
}

Tensor& Tensor::squeeze_(int dim) {

  int ndim = this->dim();

  // valid range is [-ndim, ndim)
  if (!(dim >= -ndim && dim < ndim)) {
    throw ExceptionInvalidArgument("squeeze dim out of valid range");
  }

  dim = (dim < 0 ? ndim + dim : dim);
  if (this->shape()[dim] != 1) {
    throw ExceptionInvalidArgument("squeeze dim shape not 1");
  }

  this->shape().remove(dim);
  this->strides().remove(dim);

  SHOULD_NOT_BE_CALLED_WHEN_REQUIRES_GRAD("squeeze_", *this);

  return *this;
}

Tensor& Tensor::unsqueeze_(int dim) {
  int ndim = this->dim();

  // valid range is [-ndim - 1, ndim]
  if (!(dim >= -ndim - 1 && dim <= ndim)) {
    throw ExceptionInvalidArgument("unsqueeze_ dim out of valid range");
  }

  dim = (dim < 0 ? ndim + 1 + dim : dim);

  int stride = 1;

  for (int i = dim; i < ndim; i++) {
    stride *= this->shape()[i];
  }

  this->strides().insert(dim, stride);
  this->shape().insert(dim, 1);

  SHOULD_NOT_BE_CALLED_WHEN_REQUIRES_GRAD("unsqueeze_", *this);

  return *this;
} 

Tensor Tensor::unfold(int dim, int size, int step) const {
  return toytorch::unfold(*this, dim, size, step);
}

Tensor Tensor::view(const TensorShape& new_shape) const {
  if (!is_contiguous()) {
    throw ExceptionNotImpl(
        "view() doesn't support incontiguous memory format tensor");
  }

  // Since is_contiguous is true, the product of shape dims equals to data_size()
  int total = numel();

  int minus_1_index = -1;
  int new_total = 1;
  for (int i = 0; i < new_shape.size(); i++) {
    new_total *= new_shape[i];
    if (new_shape[i] == -1) {
      if (minus_1_index == -1) {
        minus_1_index = i;
      } else {
        throw ExceptionInvalidArgument(
            "view() shape can't contain more than one -1");
      }
    }
  }

  // No -1 dim, the new_total should equal to total
  if ((minus_1_index == -1 && new_total != total) ||
      (minus_1_index != -1 && (total % new_total != 0))) {
    throw ExceptionTensorShapeIncompatible("view() new shape is incompatible");
  }

  TensorShape adjust_shape(new_shape);
  if (minus_1_index != -1) {
    adjust_shape[minus_1_index] = -(total / new_total);
  }

  TensorShape new_stride(adjust_shape.size(), 1);
  int shape_product = 1;
  for (int i = adjust_shape.size() - 1; i > 0; i--) {
    shape_product *= adjust_shape[i];
    new_stride[i - 1] = shape_product;
  }

  Tensor result(meta_copy());
  result.shape() = adjust_shape;
  result.strides() = new_stride;

  UPDATE_BACKWARD_GRAPH(result, ViewBackward, *this);

  return result;
}

Tensor Tensor::expand(const TensorShape& new_shape) const {

  if (new_shape.size() != dim()) {
    throw ExceptionTensorShapeIncompatible("expend shape incompatible");
  }

  Tensor result(meta_copy());
  for (int i = 0; i < new_shape.size(); i++) {
    if (result.shape()[i] != new_shape[i]) {
      if (result.shape()[i] != 1) {
        throw ExceptionTensorShapeIncompatible("expend shape incompatible");
      }
      result.shape()[i] = new_shape[i];
      result.strides()[i] = 0;
    }
  }

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET("expand", {*this});

  return result;
}

Tensor Tensor::pow(const Tensor& exp) const {
  return toytorch::pow(*this, exp);
}

Tensor Tensor::transpose() const {
  return toytorch::transpose(*this);
}
Tensor Tensor::transpose(int dim1, int dim2) const {
  return toytorch::transpose(*this, dim1, dim2);
}
Tensor Tensor::sum() const {
  return toytorch::sum(*this);
}
Tensor Tensor::sum(int dim, bool keep_dim) const {
  return toytorch::sum(*this, dim, keep_dim);
}
Tensor Tensor::sum(const std::vector<int>& dims, bool keep_dim) const {
  return toytorch::sum(*this, dims, keep_dim);
}

Tensor Tensor::mean() const {
  return toytorch::mean(*this);
}
Tensor Tensor::mean(int dim, bool keep_dim) const {
  return toytorch::mean(*this, dim, keep_dim);
}
Tensor Tensor::mean(const std::vector<int>& dims, bool keep_dim) const {
  return toytorch::mean(*this, dims, keep_dim);
}

Tensor Tensor::select(int dim, int index, bool keep_dim) const {
  return toytorch::select(*this, dim, index, keep_dim);
}

Tensor Tensor::slice(int dim, int start, int end) const {
  return toytorch::slice(*this, dim, start, end);
}

void Tensor::backward() {
  assert(this->requires_grad());
  autograd::backward(*this);
}

Tensor& Tensor::add_(const Tensor& other) {

  TensorHelper::elementwise_binary_op_inplace(*this, other,
                                              TensorHelper::EWBOP_ADD);

  SHOULD_NOT_BE_CALLED_WHEN_REQUIRES_GRAD("add_", *this, other);
  return *this;
}

Tensor& Tensor::sub_(const Tensor& other) {

  TensorHelper::elementwise_binary_op_inplace(*this, other,
                                              TensorHelper::EWBOP_SUB);

  SHOULD_NOT_BE_CALLED_WHEN_REQUIRES_GRAD("sub_", *this, other);
  return *this;
}

Tensor& Tensor::mul_(const Tensor& other) {

  TensorHelper::elementwise_binary_op_inplace(*this, other,
                                              TensorHelper::EWBOP_MUL);

  SHOULD_NOT_BE_CALLED_WHEN_REQUIRES_GRAD("mul_", *this, other);
  return *this;
}

Tensor& Tensor::div_(const Tensor& other) {

  TensorHelper::elementwise_binary_op_inplace(*this, other,
                                              TensorHelper::EWBOP_DIV);

  SHOULD_NOT_BE_CALLED_WHEN_REQUIRES_GRAD("div_", *this, other);
  return *this;
}

Tensor& Tensor::bernoulli_(float p) {
  BernoulliRNGenerator brg(p);

  TensorIndices indices = this->get_indices();
  TensorIndicesWalker walker(shape(), indices);

  do {
    this->at(indices) = brg();
  } while (walker.step());

  SHOULD_NOT_BE_CALLED_WHEN_REQUIRES_GRAD("bernoulli_", *this);

  return *this;
}

}  // namespace toytorch
