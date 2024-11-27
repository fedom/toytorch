
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

  int batch_shape_size = rhs.dim() - 2;
  TensorShape batch_shape(rhs.shape().begin(), rhs.shape().end() - 2);

  TensorIndices batch_indices(batch_shape.size(), 0);

  TensorShape result_shape = batch_shape;
  result_shape.push_back(col);

  Tensor result(result_shape);

  do {

    for (int i = 0; i < col; i++) {
      float v = 0;
      for (int j = 0; j < row; j++) {
        v +=
            lhs[j] * rhs.at(TensorHelper::merge_indices(batch_indices, {j, i}));
      }
      result.at(TensorHelper::merge_indices(batch_indices, {i})) = v;
    }

  } while (TensorHelper::increment_indices(batch_indices, batch_shape));

  return result;
}

Tensor matmul_md_1d(const Tensor& lhs, const Tensor& rhs) {
  assert(lhs.dim() > 1 && rhs.dim() == 1);

  int col = lhs.shape()[lhs.dim() - 1];
  int row = lhs.shape()[lhs.dim() - 2];

  if (col != rhs.shape()[0]) {
    throw ExceptionTensorShapeIncompatible();
  }

  int batch_shape_size = lhs.dim() - 2;
  TensorShape batch_shape(lhs.shape().begin(), lhs.shape().end() - 2);
  TensorIndices batch_indices(batch_shape.size(), 0);

  TensorShape result_shape = batch_shape;
  result_shape.push_back(row);

  Tensor result(result_shape);

  do {

    for (int i = 0; i < row; i++) {
      float v = 0;
      for (int j = 0; j < col; j++) {
        v +=
            lhs.at(TensorHelper::merge_indices(batch_indices, {i, j})) * rhs[j];
      }
      result.at(TensorHelper::merge_indices(batch_indices, {i})) = v;
    }

  } while (TensorHelper::increment_indices(batch_indices, batch_shape));

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
  // original tensor and modify there shape and stride for calculation.
  Tensor broadcasted_tensor_a(lhs);
  Tensor broadcasted_tensor_b(rhs);

  TensorHelper::broadcast_tensors(broadcasted_tensor_a, broadcasted_tensor_b,
                                  2);

  TensorShape result_batch_shape(broadcasted_tensor_a.shape().begin(),
                                 broadcasted_tensor_a.shape().end() - 2);
  TensorShape result_shape(result_batch_shape);
  result_shape.push_back(row_a);
  result_shape.push_back(col_b);

  TensorIndices result_batch_indices(result_batch_shape.size(), 0);
  Tensor result(result_shape);

  do {
    for (int i = 0; i < row_a; i++) {
      for (int j = 0; j < col_b; j++) {

        assert(col_a == row_b);
        float sum = 0;
        for (int k = 0; k < col_a; k++) {
          sum += broadcasted_tensor_a.at(TensorHelper::merge_indices(
                     result_batch_indices, {i, k})) *
                 broadcasted_tensor_b.at(
                     TensorHelper::merge_indices(result_batch_indices, {k, j}));
        }

        result.at(TensorHelper::merge_indices(result_batch_indices, {i, j})) =
            sum;
      }
    }
  } while (TensorHelper::increment_indices(result_batch_indices,
                                           result_batch_shape));

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


} // namespace toytorch