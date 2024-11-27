#include "nn/tensor/tensor.h"
#include "nn/exceptions/exceptions.h"
#include "nn/autograd/autograd.h"
#include "nn/autograd/backward_node_unary_op.h"
#include "nn/operations/tensor_operations.h"
#include "nn/operations/tensor_helper.h"
#include <cassert>
#include <iostream>
#include <sstream>

namespace toytorch {

using autograd::GradInfo;

Tensor Tensor::squeeze(int dim) const {
  return toytorch::squeeze(*this, dim);
}
Tensor Tensor::unsqueeze(int dim) const {
  return toytorch::unsqueeze(*this, dim);
}

Tensor Tensor::unfold(int dim, int size, int step) const {
  return toytorch::unfold(*this, dim, size, step);
}

Tensor Tensor::view(const TensorShape& new_shape) const {
  if (!is_contiguous()) {
    throw ExceptionNotImpl("view() doesn't support incontiguous memory format tensor");
  }

  // Since is_contiguous is true, the product of shape dims equals to data_size()
  int total = data_size();


  int minus_1_index = -1;
  int new_total = 1;
  for (int i = 0; i < new_shape.size(); i++) {
    new_total *= new_shape[i];
    if (new_shape[i] == -1) {
      if (minus_1_index == -1) {
        minus_1_index = i;
      } else {
        throw ExceptionInvalidArgument("view() shape can't contain more than one -1");
      }
    }
  }

  // No -1 dim, the new_total should equal to total
  if ((minus_1_index == -1 && new_total != total) || (minus_1_index != -1 && (total % new_total != 0))){
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
  BACKWARD_NOT_IMPLEMENTED_YET(expand, {*this});

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

void Tensor::backward() {
  assert(this->requires_grad());
  autograd::backward(*this);
}


void Tensor::add_(const Tensor& other) {

  TensorHelper::elementwise_binary_op_inplace(*this, other, TensorHelper::EWBOP_ADD);

  // TODO(Leo) : 
  BACKWARD_NOT_IMPLEMENTED_YET(add_, *this, other);

}

void Tensor::sub_(const Tensor& other) {

  TensorHelper::elementwise_binary_op_inplace(*this, other, TensorHelper::EWBOP_SUB);

  // TODO(Leo) : 
  BACKWARD_NOT_IMPLEMENTED_YET(sub_, *this, other);
}

void Tensor::mul_(const Tensor& other) {

  TensorHelper::elementwise_binary_op_inplace(*this, other, TensorHelper::EWBOP_MUL);

  // TODO(Leo) : 
  BACKWARD_NOT_IMPLEMENTED_YET(mul_, *this, other);

}

void Tensor::div_(const Tensor& other) {

  TensorHelper::elementwise_binary_op_inplace(*this, other, TensorHelper::EWBOP_DIV);

  // TODO(Leo) : 
  BACKWARD_NOT_IMPLEMENTED_YET(div_, *this, other);
}

}  // namespace toytorch
