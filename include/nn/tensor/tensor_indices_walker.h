#ifndef TOYTORCH_NN_TENSOR_TENSOR_INDICES_WALKER_H__
#define TOYTORCH_NN_TENSOR_TENSOR_INDICES_WALKER_H__
#include <cassert>
#include "types.h"

namespace toytorch {

#define DO_NOTHING_FOR_SCALAR() \
  do {                          \
    if (shape_.empty()) {       \
      return;                   \
    }                           \
  } while (false);

/**
 * @brief This class keep a reference to the indices and will traverse it
 * with different policies. We can create multiple walker objects to reference
 * the same indices' different parts which is very convinent for many senarios.
 */
class TensorIndicesWalker {
 public:
  TensorIndicesWalker(const TensorShape& shape, TensorIndices& indices)
      : shape_(shape), indices_(indices) {

    DO_NOTHING_FOR_SCALAR();
    // if (shape.empty()) {
    //   throw ExceptionInvalidArgument(
    //       "TensorIndicesWalker constructor failed. Can't handle scalar");
    // }
    reset_policy();
  }

  /**
   * @brief Limit the advancement to a given index at dimension dim. This is useful for
   * operations like select() where data at given index at the given dim are accessed.
   * 
   * @param dim 
   * @param index 
   */
  void narrow_to_index(int dim, int index) {
    narrow_to_index_range(dim, index, 1);
  }

  /**
   * @brief Limit the advancement to a given index range at dimension dim. This is useful
   * for operations like select() where data at given index range at the given dim are
   * accessed. Note calling this function will init the indices_[dim] to start.
   * 
   * @param dim 
   * @param start
   * @param len 
   */
  void narrow_to_index_range(int dim, int start, int len) {
    DO_NOTHING_FOR_SCALAR();
    dim = normalize_dim_index(dim);

    if (len <= 0 || start + len > shape_[dim]) {
      throw ExceptionInvalidArgument(
          "narrow_to_index_range() len exceed range");
    }

    policies_[dim].start = start;
    policies_[dim].end_included = start + len - 1;

    indices_[dim] = start;
  }

  /**
   * @brief Freeze the dim range which means when we call next(), these dims are skiped
   * and only indices in the other dims are advanced.
   * 
   * @param start_dim 
   * @param len 
   */
  void freeze_dim_range(int start, int len) {
    DO_NOTHING_FOR_SCALAR();

    start = normalize_dim_index(start);

    if (len <= 0 || start + len > shape_.size()) {
      throw ExceptionInvalidArgument("freeze_dim_range() len exceed range");
    }

    for (int i = start; i < start + len; i++) {
      policies_[i].start = -1;
      policies_[i].end_included = -1;
    }
  }

  void set_dim_stride(int dim, int stride) {
    DO_NOTHING_FOR_SCALAR();

    dim = normalize_dim_index(dim);

    if (dim >= shape_.size()) {
      throw ExceptionInvalidArgument("set_dim_stride() dim exceed range");
    }

    policies_[dim].stride = stride;
  }

  bool step() {

    for (int i = indices_.size() - 1; i >= 0; i--) {
      if (is_dim_frozen(i)) {
        continue;
      }

      assert(indices_[i] >= policies_[i].start &&
             indices_[i] <= policies_[i].end_included);

      indices_[i] += policies_[i].stride;

      if (indices_[i] <= policies_[i].end_included) {
        return true;
      }
      indices_[i] = policies_[i].start;
    }

    return false;
  }

  void reset() {
    reset_policy();
    reset_indices();
  }

  const TensorIndices& indices() const { return indices_; }
  TensorIndices& indices() { return indices_; }

 private:
  void reset_indices() {
    for (int i = 0; i < indices_.size(); i++) {
      indices_[i] = 0;
    }
  }

  void reset_policy() {
    policies_.clear();

    for (int i : shape_) {
      policies_.push_back({0, i - 1, 1});
    }
  }

  inline int normalize_dim_index(int index) const {
    return index < 0 ? indices_.size() + index : index;
  }

  inline bool is_dim_frozen(int dim) const {
    return policies_[dim].start == -1;
  }

  struct IndexRange {
    int start;
    int end_included;
    int stride;
  };

  std::vector<IndexRange> policies_;

  TensorShape shape_;
  TensorIndices& indices_;
};

}  // namespace toytorch

#endif  // TOYTORCH_NN_TENSOR_TENSOR_INDICES_WALKER_H__