
#ifndef TOYTORCH_NN_TENSOR_TENSOR_H__
#define TOYTORCH_NN_TENSOR_TENSOR_H__
#include <cassert>
#include <functional>
#include <vector>
#include "nn/autograd/tensor_grad_info.h"
#include "nn/tensor/random_generator.h"
#include "tensor_impl.h"

namespace toytorch {

class Tensor {
 public:
  Tensor() {}

  // for 0-dimensional scalar tensor
  Tensor(float n) : impl_(std::make_shared<TensorImpl>(n)) {};  

  Tensor(const TensorShape& shape, float val = 0,
         bool requires_grad = false) : impl_(std::make_shared<TensorImpl>(shape, val, requires_grad)) {}
  Tensor(const TensorShape& shape, const std::vector<float>& data,
         bool requires_grad = false) : impl_(std::make_shared<TensorImpl>(shape, data, requires_grad)) {}
  Tensor(const TensorShape& shape, RandomGeneratorBase& gen,
         bool requires_grad = false) : impl_(std::make_shared<TensorImpl>(shape, gen, requires_grad)) {}

  // Tensor(TensorImpl* impl) : impl_(impl) {}
  Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}

  virtual ~Tensor() {}

  Tensor(const Tensor& other) = default;
  Tensor(Tensor&& other) = default;
  Tensor& operator=(const Tensor& other) = default;
  Tensor& operator=(Tensor&& other) = default;


  // Access operators
  float& at(const TensorIndices& indices) { return impl_->at(indices); }
  float at(const TensorIndices& indices) const { return impl_->at(indices); }

  inline float& operator[](int index) { return (*impl_)[index]; }
  inline float operator[](int index) const { return (*impl_)[index]; }

  inline const TensorShape& shape() const { return impl_->shape(); }
  inline TensorShape& shape() { return impl_->shape(); }

  inline const TensorShape& strides() const { return impl_->strides(); }
  inline TensorShape& strides() { return impl_->strides(); }

  inline size_t dim() const { return impl_->dim(); }
  inline size_t data_size() const { return impl_->data_size(); }
  inline bool is_scalar() const { return impl_->is_scalar(); }
  inline bool is_contiguous() const {return impl_->is_contiguous();}

  std::string name() const {return "";}

  // This will only copy the meta data but share the raw data
  Tensor meta_copy() const {
    return Tensor(std::make_shared<TensorImpl>(*impl_));
  }

  Tensor deep_copy() const {
    return Tensor(std::make_shared<TensorImpl>(impl_->deep_copy()));
  }

  Tensor detach() const {
    return Tensor(std::make_shared<TensorImpl>(impl_->detach()));
  }

  void fill(const std::vector<float>& data) {
    return impl_->fill(data);
  }

  bool operator==(const Tensor& other) const {
    
    return (impl_ == other.impl_) || (*impl_) == (*other.impl_);
  }
  bool operator!=(const Tensor& other) const { return !(*this == other); }

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
  bool strict_equal(const Tensor& other) const {
    return impl_->strict_equal(*other.impl_);
  }

  bool strict_allclose(const Tensor& other, float rtol = 1e-5,
                       float atol = 1e-8, bool equal_nan = false) const {
    return impl_->strict_allclose(*other.impl_, rtol, atol, equal_nan);
  }

  void add_(const Tensor &other);
  void sub_(const Tensor &other);
  void mul_(const Tensor &other);
  void div_(const Tensor &other);

  // operation shortcut
  Tensor squeeze(int dim) const;
  Tensor unsqueeze(int dim) const;  
  Tensor unfold(int dim, int size, int step = 1) const;  
  Tensor expand(const TensorShape &shape) const;
  Tensor view(const TensorShape &shape) const;
  Tensor pow(const Tensor &exp) const;

  Tensor transpose() const;
  Tensor transpose(int dim1, int dim2) const;

  Tensor mean() const;
  Tensor mean(int dim, bool keep_dim = false) const;
  Tensor mean(const std::vector<int>& dims, bool keep_dim = false) const;

  Tensor sum() const;
  Tensor sum(int dim, bool keep_dim = false) const;
  Tensor sum(const std::vector<int>& dims, bool keep_dim = false) const;
  Tensor select(int dim, int index, bool keep_dim = false) const;

  bool requires_grad() const { return impl_->requires_grad(); }

  std::shared_ptr<autograd::GradInfo> grad_info() const {
    return impl_->grad_info();
  }
  std::shared_ptr<Tensor> grad() const { return impl_->grad(); }

  void init_grad_info() { impl_->init_grad_info(); }

  void backward();

  const float* raw_data() const { return impl_->raw_data(); }
  void print() const { impl_->print(); }

  uintptr_t identity() const {return reinterpret_cast<uintptr_t>(impl_.get());}

 private:

  std::shared_ptr<TensorImpl> impl_;
};

} // namespace toytorch

#endif // TOYTORCH_NN_TENSOR_TENSOR_H__