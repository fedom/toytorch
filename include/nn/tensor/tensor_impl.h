#ifndef SRC_NN_TENSOR_IMPL_H__
#define SRC_NN_TENSOR_IMPL_H__
#include <cassert>
#include <functional>
#include "nn/tensor/types.h"
#include "random_generator.h"
#include "nn/autograd/tensor_grad_info.h"

namespace toytorch {

namespace  autograd {
class GradInfo;
}

class TensorImpl {
 public:

  TensorImpl() {}
  TensorImpl(float n);  // for 0-dimensional scalar tensor
  TensorImpl(const TensorShape& shape, float val = 0, bool requires_grad = false);  // for multi-dimensional
  TensorImpl(const TensorShape& shape, const std::vector<float>& data, bool requires_grad = false);
  TensorImpl(const TensorShape& shape, RandomGeneratorBase& gen,
         bool requires_grad = false);

  virtual ~TensorImpl() {}

  // copy/move constructor/assignment
  TensorImpl(const TensorImpl& other);
  TensorImpl(TensorImpl&& other);
  TensorImpl& operator=(const TensorImpl& other);
  TensorImpl& operator=(TensorImpl&& other);

  TensorImpl deep_copy() const;





  // Access operators
  float& at(const TensorIndices& indices);
  float at(const TensorIndices& indices) const;

  inline float& operator[](int index) { return data_->at(index); }
  inline float operator[](int index) const { return data_->at(index); }

  // Check properties
  inline bool is_scalar() const { return dim() == 0; }
  inline bool is_contiguous() const {
    if (is_scalar()) {
      return true;
    }

    // The last stride equal 1
    if (strides_[strides_.size() - 1] != 1) {
      return false;
    }

    int cur_shape_product = 1;
    for (int i = shape_.size() - 1; i >= 1; i--) {
      cur_shape_product *= shape_[i];
      if (strides_[i - 1] != cur_shape_product) {
        return false;
      }
    }

    return true;
  }

  inline bool requires_grad() const { return !!grad_info_; }


  // Member accessors
  inline const TensorShape& shape() const { return shape_; }
  inline TensorShape& shape() { return shape_; }

  inline const TensorShape& strides() const { return strides_; }
  inline TensorShape& strides() { return strides_; }


  inline size_t dim() const { return shape_.size(); }
  inline size_t data_size() const { return data_->size(); }

  const float* raw_data() const { return data_->data(); }

  std::shared_ptr<autograd::GradInfo> grad_info() const {return grad_info_;}

  std::shared_ptr<Tensor> grad() const {
    if (grad_info_) {
      return grad_info_->grad;
    }
    return nullptr;
  }

  void init_grad_info() {
    grad_info_ = std::make_shared<autograd::GradInfo>();
    grad_info_->grad = std::make_shared<Tensor>(0);
  }
  TensorImpl detach() const;

  void fill(const std::vector<float>& data);

  // comparison operators
  bool operator==(const TensorImpl& other) const;
  bool operator!=(const TensorImpl& other) const;

  /**
   * @brief Strict equality. This checks strict element-wise equality between
   * two tensors, including their shapes and values. It compares every element
   * at the corresponding indices, and both tensors must have the same shape
   * for the comparison to return true
   * 
   * @param other 
   * @return true 
   * @return false
   */
  bool strict_equal(const TensorImpl& rhs) const;


  bool strict_allclose(const TensorImpl& rhs, float rtol = 1e-5,
                       float atol = 1e-8, bool equal_nan = false) const;



  // Autograd 
  // void backward();
  
  // TensorImpl transpose() const;
  // TensorImpl transpose(int dim1, int dim2) const;

  // TensorImpl sum() const;
  // TensorImpl sum(int dim, bool keep_dim = false) const;
  // TensorImpl sum(const std::vector<int> &dims, bool keep_dim = false) const;
  // TensorImpl take(int dim, int index, bool keep_dim = false) const;


  // Debug
  void print() const;

 private:
  std::string print_level(int base_index, int layer) const;
  // bool strict_allclose_element(float a, float b, float rtol, float atol,
  //                              bool equal_nan) const;

  /**
   * @brief Compare each element at corresponding indices separately. Return a
   * Tensor of the same shape as input with 0s and 1s. 
   * 
   * @param other 
   * @return Tensor of same shape with 0s and 1s 
   */
  // bool eq(const Tensor& other) const;

  friend class TensorHelper;

  int compute_flat_index(const TensorIndices& indices) const {
    int flat_index = 0;

    for (int i = 0; i < indices.size(); i++) {
      flat_index += strides_[i] * indices[i];
    }

    return flat_index;
  }

  // friend Tensor sum(const Tensor &tensor, const std::vector<int>& dims,
  //                  bool keep_dim /* = false*/);
  // friend Tensor transpose(const Tensor &tensor, int dim1, int dim2);

  std::shared_ptr<std::vector<float>> data_;
  TensorShape shape_;
  TensorShape strides_;

  // int offset_;

  mutable std::shared_ptr<autograd::GradInfo> grad_info_;
};

}  // namespace toytorch

#endif  // SRC_NN_TENSOR_IMPL_H__