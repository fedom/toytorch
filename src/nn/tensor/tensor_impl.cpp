#include "nn/tensor/tensor_impl.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include "nn/autograd/autograd.h"
#include "nn/exceptions/exceptions.h"
#include "nn/operations/tensor_helper.h"
#include "nn/operations/tensor_operations.h"
#include "nn/utils/print_utils.h"

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
}  // namespace

using autograd::GradInfo;

TensorImpl::TensorImpl(float n) : offset_(0) {
  data_ = std::make_shared<std::vector<float>>(1, n);
}

TensorImpl::TensorImpl(const TensorShape& shape, float val /* = 0*/,
                       bool requires_grad /* = false*/)
    : shape_(shape), strides_(shape.size(), 1), offset_(0) {
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
  for (int i = 0; i < std::min(numel(), data.size()); i++) {
    // Since this is a constructor, we guarantee the memory is contiguous
    data_->at(i) = data[i];
  }
}

TensorImpl::TensorImpl(const TensorShape& shape, RandomGeneratorBase& gen,
                       bool requires_grad /* = false*/)
    : TensorImpl(shape, 0, requires_grad) {

  // update data with random generator
  for (int i = 0; i < numel(); i++) {
    // Since this is a constructor, we guarantee the memory is contiguous
    data_->at(i) = gen();
  }
}

TensorImpl::TensorImpl(const TensorImpl& other) {
  shape_ = other.shape_;
  strides_ = other.strides_;
  data_ = other.data_;
  offset_ = other.offset_;

  grad_info_ = other.grad_info_;
}

TensorImpl::TensorImpl(TensorImpl&& other) {
  shape_ = std::move(other.shape_);
  strides_ = std::move(other.strides_);
  data_ = std::move(other.data_);
  offset_ = other.offset_;

  // requires_grad_ = other.requires_grad_;
  grad_info_ = std::move(other.grad_info_);
}

TensorImpl& TensorImpl::operator=(const TensorImpl& other) {
  shape_ = other.shape_;
  strides_ = other.strides_;
  data_ = other.data_;
  offset_ = other.offset_;

  // requires_grad_ = other.requires_grad_;
  grad_info_ = other.grad_info_;

  return *this;
}

TensorImpl& TensorImpl::operator=(TensorImpl&& other) {
  shape_ = std::move(other.shape_);
  strides_ = std::move(other.strides_);
  data_ = std::move(other.data_);
  grad_info_ = std::move(other.grad_info_);
  offset_ = other.offset_;

  return *this;
}

float& TensorImpl::at(const TensorIndices& indices) {
  int index = compute_flat_index(indices);
  return raw_data()[index];
}

float TensorImpl::at(const TensorIndices& indices) const {
  int index = compute_flat_index(indices);
  return raw_data()[index];
}

TensorImpl TensorImpl::deep_copy() const {
  TensorImpl result(shape_);

  // Deep copy doesn't copy grad_info_. Instead, we shared the instance
  // with original one.
  result.grad_info_ = grad_info_;

  if (is_contiguous()) {
    std::copy(raw_data(), raw_data() + numel(), result.raw_data());
    return result;
  }

  TensorIndices indices(dim(), 0);
  int i = 0;
  do {
    // result is guaranteed to be contiguous
    result[i++] = this->at(indices);
  } while (TensorHelper::increment_indices(indices, shape_));

  // TODO(Leo) : we should insert this operation into backward graph

  return result;
}

TensorImpl TensorImpl::detach() const {
  TensorImpl result(*this);
  result.grad_info_.reset();

  return result;
}

void TensorImpl::fill(const std::vector<float>& data) {
  if (numel() != data.size()) {
    throw ExceptionInvalidArgument("Data size doesn't match with tensor size");
  }

  if (is_contiguous()) {
    std::copy(data.begin(), data.end(), raw_data());
    return;
    //throw ExceptionNotImpl("fill for uncontinuous memory format not supported yet");
  }

  TensorIndices indices(dim(), 0);
  int i = 0;
  do {
    this->at(indices) = data.at(i++);
  } while (TensorHelper::increment_indices(indices, shape()));
}

bool TensorImpl::strict_equal(const TensorImpl& rhs) const {
  if (shape() != rhs.shape()) {
    return false;
  }
  // For scalars
  if (dim() == 0) {
    return (*this)[offset_] == rhs[rhs.offset_];
  }

  TensorIndices indices(dim(), 0);

  do {
    if (at(indices) != rhs.at(indices)) {
      return false;
    }

  } while (TensorHelper::increment_indices(indices, shape()));

  return true;
}

bool TensorImpl::strict_allclose(const TensorImpl& rhs, float rtol, float atol,
                                 bool equal_nan) const {
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

void TensorImpl::print() const {
  if (is_scalar()) {
    std::cout << *(raw_data()) << std::endl;
    return;
  }
  std::cout << print_level(offset_, 0) << std::endl;
}

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
  std::cout << shape_ << std::endl;
}

void TensorImpl::print_strides() const {
  std::cout << strides_ << std::endl;
}

}  // namespace toytorch
