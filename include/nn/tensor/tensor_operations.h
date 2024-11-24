#ifndef TOYTORCH_NN_TENSOR_TENSOR_OPERATIONS_H__
#define TOYTORCH_NN_TENSOR_TENSOR_OPERATIONS_H__
#include "tensor.h"

namespace toytorch {

enum class ReductionType {
  None = 0,
  Mean,
  Sum,
};

/**
   * Policies for matmal. There are several situations to consider
   *  1. 1-D & 1-D : shape (m,) & (n)
   *    (1) if m != n throw else do dot product,
   *    (2) result is a scalar tensor
   * 
   *  2. 1-D & 2-D : shape (m,) & (n, p)
   *    (1) if m != n throw else do vector * matrix
   *    (2) result shape is (p,)
   * 
   *  3. 1-D & multi-D : shape (m,) & (k, l, n, p)
   *    (1) if m != n throw else do vector * matrix (last 2 dimensions), k, l as batch dimension
   *    (2) result shape is (k, l, p)
   * 
   *  4. 2-D & 1-D : shape (m, n) & (p, )
   *    (1) if n != p throw else do matrix * vector
   *    (2) result shape is (m, ) 
   * 
   *  5. 2-D & 2-D : shape (m, n) & (p, q)
   *    (1) if n != p  throw else do matrix multiplication
   *    (2) result shape is (m, q)
   * 
   *  6. 2-D & Multi-D shape (m, n) & (k, l, p, q)
   *    (1) if n != p throw else do matrix multiplication(last 2 dimensions), k, l as batch dimensions
   *    (2) result shape is (k, l, m, q)
   * 
   *  7. Multi-D & 1-D : shape (k, l, m, n) & (p, )
   *    (1) if n != p throw else do matrix * vector , k, l as batch dimensions
   *    (2) result shape is (k, l, m)
   * 
   *  8. Multi-D & 2-D : shape (k, l, m, n) & (p, q)
   *    (1) if n != p throw else do matrix multiplication
   *    (2) result shape is (k, l, m, q)
   * 
   *  9. Multi-D & Multi-D : ...
   * 
   *  When implement this, we can categorize them into two categories:
   *  (1) separate logic for each case: 1, 2, 3, 4, 7 (operand includes 1-D tensor)
   *  (2) implement them into one generalized logic: 5, 6, 8, 9
   *
   * @param lhs The first tensor.
   * @param rhs The second tensor.
   * @return The result of matmul.
   */
Tensor matmul(const Tensor& lhs, const Tensor& rhs);

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

Tensor sigmoid(const Tensor& tensor);
Tensor relu(const Tensor& tensor);

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

// Loss functions
Tensor smooth_l1_loss(const Tensor& input, const Tensor& target,
                      ReductionType rt = ReductionType::Mean, float beta = 1.0);
Tensor mse_loss(const Tensor& input, const Tensor& target);

}  // namespace toytorch

#endif  // TOYTORCH_NN_TENSOR_TENSOR_OPERATIONS_H__