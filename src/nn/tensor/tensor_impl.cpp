#include "nn/tensor/tensor_impl.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include "exception/exceptions.h"
#include "nn/autograd/autograd.h"
#include "nn/tensor/tensor_operations.h"
#include "nn/tensor/tensor_operations.h"
#include "tensor_helper.h"

namespace toytorch {

namespace {
  bool strict_allclose_element(float a, float b, float rtol, float atol,
                             bool equal_nan) {

  if (std::isnan(a) || std::isnan(b)) {
    if (std::isnan(a) && std::isnan(b) && equal_nan) {
      return true;
    }
    return false;
  }

  return std::abs(a - b) <= atol + rtol * std::abs(b);
}
}

using autograd::GradInfo;

TensorImpl::TensorImpl(float n) {
  data_ = std::make_shared<std::vector<float>>(1, n);
}

TensorImpl::TensorImpl(const TensorShape& shape, float val /* = 0*/,
               bool requires_grad /* = false*/)
    : shape_(shape), strides_(shape.size(), 1) {
  int size = 1;

  int accum_stride = 1;
  for (int i = shape.size() - 1; i >= 0; i--) {
    size *= shape[i];
    strides_[i] = accum_stride;
    accum_stride = shape[i] * accum_stride;
  }

  data_ = std::make_shared<std::vector<float>>(size, val);

  if (requires_grad) {
    init_grad_info();
  }
}

TensorImpl::TensorImpl(const TensorShape& shape, const std::vector<float>& data,
               bool requires_grad)
    : TensorImpl(shape, 0, requires_grad) {

  // update data with vector data
  for (int i = 0; i < std::min(data_size(), data.size()); i++) {
    data_->at(i) = data[i];
  }
}

TensorImpl::TensorImpl(const TensorShape& shape, RandomGeneratorBase& gen,
               bool requires_grad /* = false*/)
    : TensorImpl(shape, 0, requires_grad) {

  // update data with random generator
  for (int i = 0; i < data_size(); i++) {
    data_->at(i) = gen();
  }
}

TensorImpl::TensorImpl(const TensorImpl& other) {
  shape_ = other.shape_;
  strides_ = other.strides_;
  data_ = other.data_;
  // requires_grad_ = other.requires_grad_;
  grad_info_ = other.grad_info_;
}

TensorImpl::TensorImpl(TensorImpl&& other) {
  shape_ = std::move(other.shape_);
  strides_ = std::move(other.strides_);
  data_ = std::move(other.data_);

  // requires_grad_ = other.requires_grad_;
  grad_info_ = std::move(other.grad_info_);
}

TensorImpl& TensorImpl::operator=(const TensorImpl& other) {
  shape_ = other.shape_;
  strides_ = other.strides_;
  data_ = other.data_;

  // requires_grad_ = other.requires_grad_;
  grad_info_ = other.grad_info_;

  return *this;
}

TensorImpl& TensorImpl::operator=(TensorImpl&& other) {
  shape_ = std::move(other.shape_);
  strides_ = std::move(other.strides_);
  data_ = std::move(other.data_);
  grad_info_ = std::move(other.grad_info_);
  // requires_grad_ = other.requires_grad_;

  return *this;
}

float& TensorImpl::at(const TensorIndices& indices) {
  int index = compute_flat_index(indices);
  return data_->at(index);
}

float TensorImpl::at(const TensorIndices& indices) const {
  int index = compute_flat_index(indices);
  return data_->at(index);
}

TensorImpl TensorImpl::deep_copy() const {
  TensorImpl result(*this);

  result.data_ = std::make_shared<std::vector<float>>(*data_);

  // Deep copy doesn't copy grad_info_. Instead, we shared the instance
  // with original one.
  result.grad_info_ = grad_info_;

  return result;
}

TensorImpl TensorImpl::detach() const {
  TensorImpl result(*this);
  result.grad_info_.reset();

  return result;
}

void TensorImpl::fill(const std::vector<float>& data) {
  if (data_size() != data.size()) {
    throw ExceptionInvalidArgument("Data size doesn't match with tensor size");
  }

  if (!is_contiguous()) {
    throw ExceptionNotImpl("fill for uncontinuous memory format not supported yet");
  }
  std::copy(data.begin(), data.end(), data_->begin());
}

bool TensorImpl::strict_equal(const TensorImpl& rhs) const {
  if (shape() != rhs.shape()) {
    return false;
  }
  // For scalars
  if (dim() == 0) {
    return (*this)[0] == rhs[0];
  }

  TensorIndices indices(dim(), 0);

  do {
    if (at(indices) != rhs.at(indices)) {
      return false;
    }

  } while (TensorHelper::increment_indices(indices, shape()));

  return true;
}

bool TensorImpl::strict_allclose(const TensorImpl& rhs, float rtol,
                     float atol, bool equal_nan) const {
  if (shape() != rhs.shape()) {
    return false;
  }

  // For scalars
  if (dim() == 0) {
    strict_allclose_element((*this)[0], rhs[0], rtol, atol, equal_nan);
  }

  TensorIndices indices(dim(), 0);

  do {
    if (!strict_allclose_element(at(indices), rhs.at(indices), rtol, atol,
                                 equal_nan)) {
      return false;
    }

  } while (TensorHelper::increment_indices(indices, shape()));

  return true;
}

bool TensorImpl::operator==(const TensorImpl& other) const {
  return strict_equal(other);
}

bool TensorImpl::operator!=(const TensorImpl& other) const {
  return !(*this == other);
}

// Tensor Tensor::transpose() const {
//   return toynn::transpose(*this);
// }
// Tensor Tensor::transpose(int dim1, int dim2) const {
//   return toynn::transpose(*this, dim1, dim2);
// }
// Tensor Tensor::sum() const {
//   return toynn::sum(*this);
// }
// Tensor Tensor::sum(int dim, bool keep_dim) const {
//   return toynn::sum(*this, dim, keep_dim);
// }
// Tensor Tensor::sum(const std::vector<int>& dims, bool keep_dim) const {
//   return toynn::sum(*this, dims, keep_dim);
// }
// Tensor Tensor::take(int dim, int index, bool keep_dim) const {
//   return toynn::take(*this, dim, index, keep_dim);
// }

// void TensorImpl::backward() {
//   assert(this->requires_grad());
//   autograd::backward(*this);
// }

void TensorImpl::print() const {
  if (is_scalar()) {
    std::cout << *(raw_data()) << std::endl;
    return;
  }
  std::cout << print_level(0, 0) << std::endl;
}

// std::string Tensor::print_level(int flat_index_base, int layer) const {

//   if (layer == dim()) {  // last layer
//     return std::to_string(data_->at(flat_index_base));
//   }

//   // intermediate layers
//   std::ostringstream ss;
//   ss << "[";
//   std::string sep = "";
//   for (int i = 0; i < shape_[layer]; i++) {
//     ss << sep;
//     ss << print_level(flat_index_base + strides_[layer] * i, layer + 1);
//     sep = ",";
//   }
//   ss << "]";
//   return ss.str();
// }

std::string TensorImpl::print_level(int flat_index_base, int layer) const {

  if (layer == dim() - 1) {  // last layer
    std::ostringstream ss;
    ss << "[";
    std::string sep = "";
    for (int i = 0; i < shape_[layer]; i++) {
      ss << sep;
      ss << data_->at(flat_index_base + strides_[layer] * i);
      sep = ",";
    }
    ss << "]";
    return ss.str();
  }

  // intermediate layers
  std::ostringstream ss;
  ss << "[";
  std::string sep = "";
  for (int i = 0; i < shape_[layer]; i++) {
    ss << sep;
    ss << print_level(flat_index_base + strides_[layer] * i, layer + 1);
    sep = ",\n";
  }
  ss << "]";
  return ss.str();
}

void TensorImpl::print_shape() const {
  std::ostringstream oss;

  oss << "[";
  for (auto i : shape_) {
    oss << i << ",";
  }
  oss << "]";
  std::cout << oss.str() << std::endl;
}


}  // namespace toytorch
