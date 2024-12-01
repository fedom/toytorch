#ifndef TOYTORCH_NN_UTILS_EXTENDED_VECTOR_H__
#define TOYTORCH_NN_UTILS_EXTENDED_VECTOR_H__
#include <vector>
#include <cassert>
#include "nn/exceptions/exceptions.h"

namespace toytorch {

template <typename T>
class ExtendedVector {
 public:
  ExtendedVector() = default;
  ExtendedVector(const std::initializer_list<T>& values) : values_(values) {}
  ExtendedVector(std::vector<T> values) : values_(std::move(values)) {}
  ExtendedVector(size_t size, const T& value = T()) : values_(size, value) {}

  bool operator==(const ExtendedVector& other) const {
    return values_ == other.values_;
  }

  bool operator!=(const ExtendedVector& other) const {
    return !(*this == other);
  }

  T at(int index) const {
    index = normalize_index(index);
    return values_.at(index);
  }

  T& at(int index) {
    index = normalize_index(index);
    return values_.at(index);
  }

  T operator[](int index) const { return at(index); }

  T& operator[](int index) { return at(index); }

  size_t size() const { return values_.size(); }
  bool empty() const { return values_.empty(); }

  ExtendedVector copy() const {
    return *this;
  }

  ExtendedVector subvector(int start, int len = -1) const {
    if (len < 0) {
      return ExtendedVector(
          std::vector<T>(values_.begin() + start, values_.end()));
    }

    if (start + len > values_.size()) {
      throw ExceptionInvalidArgument("ExtendedVector::subvector len out of range");
    }

    return ExtendedVector(
        std::vector<T>(values_.begin() + start, values_.begin() + start + len));
  }

  ExtendedVector expand_left_to_copy(int to_size, int val = 1) const {
    ExtendedVector result(*this);
    result.expand_left_to(to_size, val);
    return result;
  }

  ExtendedVector& expand_left_to(int to_size, int val = 1) {
    if (size() == to_size)
      return *this;

    values_.insert(values_.begin(), to_size - size(), val);
    return *this;
  }

  // Iterators
  auto begin() { return values_.begin(); }
  auto end() { return values_.end(); }
  auto begin() const { return values_.cbegin(); }
  auto end() const { return values_.cend(); }

  void insert(int dim, int val) {
    dim = normalize_index(dim);
    values_.insert(values_.begin() + dim, val);
  }

  ExtendedVector insert_copy(int dim, int val) const {
    ExtendedVector result(*this);
    result.insert(dim, val);

    return result;
  }

  void remove(int dim) {
    dim = normalize_index(dim);
    values_.erase(values_.begin() + dim);
  }

  ExtendedVector remove_copy(int dim) const {
    ExtendedVector result(*this);
    result.remove(dim);
    return result;
  }

  ExtendedVector& push_back(int val) {
    values_.push_back(val);
    return *this;
  }

  ExtendedVector push_back_copy(int val) const {
    ExtendedVector result(*this);
    result.push_back(val);
    return result;
  }

  auto split2(int dim) const -> std::tuple<ExtendedVector, ExtendedVector> {
    dim = normalize_index(dim);

    assert(dim >= 0 && dim < values_.size());

    ExtendedVector left_part(
        std::vector<int>(values_.begin(), values_.begin() + dim));
    ExtendedVector right_part(
        std::vector<int>(values_.begin() + dim, values_.end()));

    return std::forward_as_tuple(left_part, right_part);
  }

  auto split3(int dim) const
      -> std::tuple<ExtendedVector, ExtendedVector, ExtendedVector> {

    dim = normalize_index(dim);

    assert(dim >= 0 && dim < values_.size());

    ExtendedVector left_part(
        std::vector<int>(values_.begin(), values_.begin() + dim));
    ExtendedVector middle_part(
        std::vector<int>(values_.begin() + dim, values_.begin() + dim + 1));
    ExtendedVector right_part(
        std::vector<int>(values_.begin() + dim + 1, values_.end()));

    return std::forward_as_tuple(left_part, middle_part, right_part);
  }

  ExtendedVector concat_copy(const ExtendedVector& other) const {
    ExtendedVector combined(*this);
    combined.values_.insert(combined.values_.end(), other.values_.begin(),
                            other.values_.end());
    return combined;
  }

  ExtendedVector& concat(const ExtendedVector& other) {
    values_.insert(values_.end(), other.values_.begin(), other.values_.end());
    return *this;
  }

 private:
  inline int normalize_index(int index) const {
    return index < 0 ? (values_.size() + index) : index;
  }

  std::vector<T> values_;
};

}  // namespace toytorch

#endif  // TOYTORCH_NN_UTILS_EXTENDED_VECTOR_H__