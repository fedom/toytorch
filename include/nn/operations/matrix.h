#ifndef TOYTORCH_NN_OPERATIONS_MATRIX_H__
#define TOYTORCH_NN_OPERATIONS_MATRIX_H__
#include "nn/tensor/tensor.h"

namespace toytorch {


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

} // namespace toytorch

#endif // TOYTORCH_NN_OPERATIONS_MATRIX_H__