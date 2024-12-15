#ifndef TOYTORCH_NN_TENSOR_TENSOR_OPERATIONS_H__
#define TOYTORCH_NN_TENSOR_TENSOR_OPERATIONS_H__
#include <array>
#include <vector>
#include "nn/tensor/tensor.h"

namespace toytorch {

inline int normalize_dim(const Tensor& t, int dim) {
  return dim < 0 ? (t.dim() + dim) : dim;
}

// Ternary ops
Tensor where(const Tensor& condition, const Tensor& input, const Tensor& other);

// Binary ops
Tensor add(const Tensor& self, const Tensor& other);
Tensor sub(const Tensor& self, const Tensor& other);
Tensor mul(const Tensor& self, const Tensor& other);
Tensor div(const Tensor& self, const Tensor& other);
Tensor pow(const Tensor& base, const Tensor& exp);
Tensor gt(const Tensor& self, const Tensor& other);
Tensor ge(const Tensor& self, const Tensor& other);
Tensor lt(const Tensor& self, const Tensor& other);
Tensor le(const Tensor& self, const Tensor& other);

Tensor operator+(const Tensor& lhs, const Tensor& rhs);
Tensor operator-(const Tensor& lhs, const Tensor& rhs);
Tensor operator*(const Tensor& lhs, const Tensor& rhs);
Tensor operator/(const Tensor& lhs, const Tensor& rhs);
Tensor operator^(const Tensor& lhs, const Tensor& rhs);
Tensor operator>(const Tensor& lhs, const Tensor& rhs);
Tensor operator>=(const Tensor& lhs, const Tensor& rhs);
Tensor operator<(const Tensor& lhs, const Tensor& rhs);
Tensor operator<=(const Tensor& lhs, const Tensor& rhs);

// Unary ops
Tensor exp(const Tensor& tensor);
Tensor neg(const Tensor& tensor);
Tensor abs(const Tensor& tensor);
Tensor sign(const Tensor& tensor);
Tensor log(const Tensor& tensor);

// sum() series will return a totally new tensor with its own raw data
Tensor sum(const Tensor& tensor);
Tensor sum(const Tensor& tensor, int dim, bool keep_dim = false);
Tensor sum(const Tensor& tensor, const std::vector<int>& dims,
           bool keep_dim = false);

Tensor mean(const Tensor& tensor);
Tensor mean(const Tensor& tensor, int dim, bool keep_dim = false);
Tensor mean(const Tensor& tensor, const std::vector<int>& dims,
            bool keep_dim = false);

// select() will return a totally new tensor with its own raw data
Tensor select(const Tensor& tensor, int dim, int index, bool keep_dim = false);

// This a shortcut for transpose(tensor, 0, 1)
Tensor transpose(const Tensor& tensor);
Tensor transpose(const Tensor& tensor, int dim1, int dim2);

Tensor cat(const std::vector<Tensor>& tensors, int dim);
Tensor squeeze(const Tensor& tensor, int dim);
Tensor unsqueeze(const Tensor& tensor, int dim);
Tensor reshape(const Tensor& tensor, const TensorShape& shape);
Tensor unfold(const Tensor& tensor, int dim, int size, int step = 1);

Tensor bernoulli(const Tensor& p);

/**
 * @brief Similar to pytorch's slice function [:,:,:]
 * 
 * @param tensor 
 * @param dim 
 * @param start 
 * @param end End index not included in the result. Pass size() of the dimension if you
 *  want include the last element
 * 
 * @return Tensor 
 */
Tensor slice(const Tensor& tensor, int dim, int start, int end);
Tensor flip(const Tensor& tensor, const std::vector<int>& dims);

Tensor max_pool2d(const Tensor& input, const std::array<int, 2>& kernel_size,
                  const std::array<int, 2>& strides);



}  // namespace toytorch

#endif  // TOYTORCH_NN_TENSOR_TENSOR_OPERATIONS_H__