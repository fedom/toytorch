
#include "nn/operations/matrix.h"
#include "nn/autograd/autograd.h"
#include "nn/autograd/backward_node_binary_op.h"
#include "nn/exceptions/exceptions.h"
#include "nn/operations/tensor_helper.h"

namespace toytorch {

using autograd::Edge;
using autograd::Node;

namespace {

Tensor matmul_1d(const Tensor& lhs, const Tensor& rhs);
Tensor matmul_1d_1d(const Tensor& lhs, const Tensor& rhs);
Tensor matmul_1d_md(const Tensor& lhs, const Tensor& rhs);
Tensor matmul_md_1d(const Tensor& lhs, const Tensor& rhs);
Tensor matmul_md(const Tensor& lhs, const Tensor& rhs);

/**
 * @brief Calculate matmul that either operand is 1d tensor. The special thing for 1d is that
 *  we should not broadcast for 1d involved matmul.
 * 
 * @param other 
 * @return Tensor 
 */
Tensor matmul_1d(const Tensor& lhs, const Tensor& rhs) {
  if (lhs.dim() == 1 && rhs.dim() == 1) {
    return matmul_1d_1d(lhs, rhs);
  }

  if (lhs.dim() == 1) {
    return matmul_1d_md(lhs, rhs);
  }

  return matmul_md_1d(lhs, rhs);
}

Tensor matmul_1d_1d(const Tensor& lhs, const Tensor& rhs) {
  if (lhs.shape()[0] != rhs.shape()[0]) {
    throw ExceptionTensorShapeIncompatible();
  }

  float r = 0;
  for (int i = 0; i < lhs.shape()[0]; i++) {
    r = r + lhs[i] * rhs[i];
  }

  return Tensor(r);
}

Tensor matmul_1d_md(const Tensor& lhs, const Tensor& rhs) {
  assert(lhs.dim() == 1 && rhs.dim() > 1);

  int col = rhs.shape()[rhs.dim() - 1];
  int row = rhs.shape()[rhs.dim() - 2];
  if (lhs.shape()[0] != row) {
    throw ExceptionTensorShapeIncompatible();
  }

  TensorShape result_shape = rhs.shape().remove_copy(-2);
  Tensor result(result_shape);

  TensorIndices result_indices = result.get_indices();
  TensorIndicesWalker result_walker(result_shape, result_indices);

  do {
    float sum = 0;

    auto&& [left, right] = result_indices.split2(-1);
    for (int i = 0; i < row; i++) {
      sum += lhs.at_raw(i) * rhs.at(left.push_back_copy(i).concat(right));
    }
    result.at(result_indices) = sum;
  } while (result_walker.step());

  return result;
}

Tensor matmul_md_1d(const Tensor& lhs, const Tensor& rhs) {
  assert(lhs.dim() > 1 && rhs.dim() == 1);

  int col = lhs.shape()[-1];
  int row = lhs.shape()[-2];

  if (col != rhs.shape()[0]) {
    throw ExceptionTensorShapeIncompatible();
  }

  TensorShape result_shape = lhs.shape().remove_copy(-1);
  Tensor result(result_shape);
  TensorIndices result_indices = result.get_indices();

  TensorIndicesWalker result_walker(result_shape, result_indices);

  do {
    float sum = 0;

    for (int i = 0; i < col; i++) {
      sum += rhs.at_raw(i) * lhs.at(result_indices.push_back_copy(i));
    }
    result.at(result_indices) = sum;

  } while (result_walker.step());

  return result;
}

Tensor matmul_md(const Tensor& lhs, const Tensor& rhs) {
  assert(lhs.dim() >= 2 && rhs.dim() >= 2);

  int row_a = lhs.shape()[lhs.dim() - 2];
  int col_a = lhs.shape()[lhs.dim() - 1];
  int row_b = rhs.shape()[rhs.dim() - 2];
  int col_b = rhs.shape()[rhs.dim() - 1];

  if (col_a != row_b || !TensorHelper::are_tensors_broadcastable(lhs, rhs, 2)) {
    throw ExceptionTensorShapeIncompatible();
  }

  // Tensor copy is a shallow copy. The copy shares the raw data, so we can copy
  // original tensor and modify their shape and stride for calculation.
  Tensor broadcasted_tensor_a(lhs.meta_copy());
  Tensor broadcasted_tensor_b(rhs.meta_copy());

  TensorHelper::broadcast_tensors(broadcasted_tensor_a, broadcasted_tensor_b,
                                  2);

  TensorShape result_shape =
      broadcasted_tensor_a.shape().remove_copy(-1).push_back(rhs.shape()[-1]);
  Tensor result(result_shape);

  TensorIndices result_indices = result.get_indices();
  TensorIndicesWalker result_walker(result_shape, result_indices);

  do {
    // lhs shape :  [A, B , C, m, k]
    // rhs shape :  [A, B , C, k, n]
    // result shape:[A, B , C, m, n]
    //
    // result shape split3: [A, B ,C , m, n] -> [A, B, C], [m], [n]
    // lhs: [A, B, C] + [m] + k
    // rhs: [A, B, C] + k + [n]
    auto&& [left, mid, right] = result_indices.split3(-2);
    float sum = 0;
    for (int i = 0; i < col_a; i++) {
      // Note here, for indices operation, we use the copy version to return a new copy so that
      // that don't interfere with the other operand's indices
      sum += broadcasted_tensor_a.at(left.concat_copy(mid).push_back(i)) *
             broadcasted_tensor_b.at(left.push_back_copy(i).concat(right));
    }
    result.at(result_indices) = sum;

  } while (result_walker.step());

  return result;
}

}  // namespace

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
Tensor matmul(const Tensor& lhs, const Tensor& rhs) {

  if (lhs.dim() == 0 || rhs.dim() == 0) {
    throw ExceptionInvalidArgument("arguments of matmul() can't have 0 dims");
  }

  Tensor result;
  if (lhs.dim() == 1 || rhs.dim() == 1) {
    result = matmul_1d(lhs, rhs);
  } else {
    result = matmul_md(lhs, rhs);
  }

  UPDATE_BACKWARD_GRAPH(result, MatmulBackward, lhs, rhs);
  return result;
}

}  // namespace toytorch