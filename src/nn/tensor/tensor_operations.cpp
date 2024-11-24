#include "nn/tensor/tensor_operations.h"
#include "exception/exceptions.h"
#include "nn/autograd/autograd.h"
#include "nn/autograd/backward_node_activation_op.h"
#include "nn/autograd/backward_node_binary_op.h"
#include "nn/autograd/backward_node_leaf_op.h"
#include "nn/autograd/backward_node_unary_op.h"
#include "nn/tensor/tensor_helper.h"

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

Tensor where(const Tensor& condition, const Tensor& input,
             const Tensor& other) {
  Tensor result = TensorHelper::elementwise_ternary_op(
      condition, input, other, TensorHelper::EWTOP_WHERE);

  // condition won't be add to the grad since there is no defined grad for it
  // We only pass it to WhereBackward() to be stored for calculation of input
  // and other's grad
  UPDATE_BACKWARD_GRAPH_1(result, WhereBackward, condition, input, other);
  return result;
}

Tensor add(const Tensor& self, const Tensor& other) {
  Tensor result =
      TensorHelper::elementwise_binary_op(self, other, TensorHelper::EWBOP_ADD);

  UPDATE_BACKWARD_GRAPH(result, AddBackward, self, other);
  return result;
}

Tensor sub(const Tensor& self, const Tensor& other) {
  Tensor result =
      TensorHelper::elementwise_binary_op(self, other, TensorHelper::EWBOP_SUB);

  UPDATE_BACKWARD_GRAPH(result, SubBackward, self, other);
  return result;
}

/**
 * @brief Do Hadamard product
 * 
 * @param other 
 * @return Tensor 
 */
Tensor mul(const Tensor& self, const Tensor& other) {
  Tensor result =
      TensorHelper::elementwise_binary_op(self, other, TensorHelper::EWBOP_MUL);
  UPDATE_BACKWARD_GRAPH(result, MulBackward, self, other);
  return result;
}

Tensor div(const Tensor& self, const Tensor& other) {
  Tensor result =
      TensorHelper::elementwise_binary_op(self, other, TensorHelper::EWBOP_DIV);
  UPDATE_BACKWARD_GRAPH(result, DivBackward, self, other);
  return result;
}

Tensor pow(const Tensor& self, const Tensor& other) {
  Tensor result =
      TensorHelper::elementwise_binary_op(self, other, TensorHelper::EWBOP_POW);
  UPDATE_BACKWARD_GRAPH(result, PowBackward, self, other);
  return result;
}

Tensor exp(const Tensor& tensor) {
  Tensor result =
      TensorHelper::elementwise_unary_op(tensor, TensorHelper::EWUOP_EXP);
  UPDATE_BACKWARD_GRAPH(result, ExpBackward, tensor);
  return result;
}

Tensor neg(const Tensor& tensor) {
  Tensor result =
      TensorHelper::elementwise_unary_op(tensor, TensorHelper::EWUOP_NEG);
  UPDATE_BACKWARD_GRAPH(result, NegBackward, tensor);
  return result;
}

Tensor abs(const Tensor& tensor) {
  Tensor result =
      TensorHelper::elementwise_unary_op(tensor, TensorHelper::EWUOP_ABS);
  UPDATE_BACKWARD_GRAPH(result, AbsBackward, tensor);
  return result;
}

Tensor sign(const Tensor &tensor) {
  Tensor result =
      TensorHelper::elementwise_unary_op(tensor, TensorHelper::EWUOP_SIGN);

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET(sign, tensor);
  
  return result;
}


Tensor unsqueeze(const Tensor& tensor, int dim) {
  int ndim = tensor.dim();
  // valid range is [-ndim - 1, ndim]
  if (!(dim >= -ndim - 1 && dim <= ndim)) {
    throw ExceptionInvalidArgument("unsqueeze dim out of valid range");
  }

  dim = (dim < 0 ? ndim + 1 + dim : dim);

  // since we need to modify the meta data, we need a meta_copy
  Tensor result(tensor.meta_copy());

  int stride = 1;

  for (int i = dim; i < ndim; i++) {
    stride *= tensor.shape()[i];
  }

  result.strides().insert(result.strides().begin() + dim, stride);
  result.shape().insert(result.shape().begin() + dim, 1);

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET(unsqueeze, tensor);

  return result;
}

Tensor squeeze(const Tensor& tensor, int dim) {

  int ndim = tensor.dim();
  // valid range is [-ndim, ndim)
  if (!(dim >= -ndim && dim < ndim)) {
    throw ExceptionInvalidArgument("squeeze dim out of valid range");
  }

  dim = (dim < 0 ? ndim + dim : dim);
  if (tensor.shape()[dim] != 1) {
    throw ExceptionInvalidArgument("squeeze dim shape not 1");
  }

  Tensor result(tensor.meta_copy());

  result.shape().erase(result.shape().begin() + dim);
  result.strides().erase(result.strides().begin() + dim);

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET(squeeze, tensor);

  return result;
}

Tensor cat(const std::vector<Tensor>& tensors, int dim) {
  if (tensors.size() < 2) {
    throw ExceptionInvalidArgument("cat() tensors count less than 2");
  }
  int ndim = tensors[0].dim();

  // dim's valid range is [-ndim, ndim - 1]
  if (!(dim < ndim && dim >= -ndim)) {
    throw ExceptionInvalidArgument("cat() arg dim out of range");
  }

  // convert to positive index
  dim = (dim < 0 ? ndim + dim : dim);

  TensorShape result_shape = tensors[0].shape();
  // check shapes are compatible
  for (int i = 1; i < tensors.size(); i++) {
    if (tensors[i].dim() != ndim) {
      throw ExceptionTensorShapeIncompatible(
          "cat() tensors has imcompatible shapes");
    }
    for (int j = 0; j < tensors[0].dim(); j++) {
      if (j == dim) {
        result_shape[j] += tensors[i].shape()[j];
        continue;
      }
      if (tensors[i].shape()[j] != tensors[0].shape()[j]) {
        throw ExceptionTensorShapeIncompatible(
            "cat() tensors has imcompatible shapes");
      }
    }
  }

  TensorShape shape_first_part;
  for (int i = 0; i < dim; i++) {
    shape_first_part.push_back(tensors[0].shape()[i]);
  }

  Tensor result(result_shape);

  TensorIndices result_indices(result_shape.size(), 0);
  TensorIndices indices_first_part(dim, 0);

  do {
    for (auto& tensor : tensors) {
      TensorShape read_shape_second_part(tensor.shape().begin() + dim,
                                         tensor.shape().end());
      TensorIndices read_indices_second_part(read_shape_second_part.size(), 0);

      do {
        TensorIndices reading_indices = TensorHelper::merge_indices(
            indices_first_part, read_indices_second_part);

        result.at(result_indices) = tensor.at(reading_indices);

        TensorHelper::increment_indices(result_indices, result_shape);

      } while (TensorHelper::increment_indices(read_indices_second_part,
                                               read_shape_second_part));
    }

  } while (
      TensorHelper::increment_indices(indices_first_part, shape_first_part));

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET_VEC(cat, tensors);

  return result;
}

// Tensor reshape(const Tensor& tensor, const TensorShape& shape) {}

Tensor gt(const Tensor& self, const Tensor& other) {
  // There is no derivative for gt() operation. The result tensor will simply
  // be a tensor with requires_grad = false.
  return TensorHelper::elementwise_binary_op(self, other,
                                             TensorHelper::EWBOP_GT);
}

Tensor lt(const Tensor& self, const Tensor& other) {
  // There is no derivative for lt() operation. The result tensor will simply
  // be a tensor with requires_grad = false.
  return TensorHelper::elementwise_binary_op(self, other,
                                             TensorHelper::EWBOP_LT);
}

Tensor ge(const Tensor& self, const Tensor& other) {
  // There is no derivative for ge() operation. The result tensor will simply
  // be a tensor with requires_grad = false.
  return TensorHelper::elementwise_binary_op(self, other,
                                             TensorHelper::EWBOP_GE);
}

Tensor le(const Tensor& self, const Tensor& other) {
  // There is no derivative for le() operation. The result tensor will simply
  // be a tensor with requires_grad = false.
  return TensorHelper::elementwise_binary_op(self, other,
                                             TensorHelper::EWBOP_LE);
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
  return add(lhs, rhs);
}
Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
  return sub(lhs, rhs);
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
  return mul(lhs, rhs);
}

Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
  return div(lhs, rhs);
}

Tensor operator^(const Tensor& lhs, const Tensor& rhs) {
  return pow(lhs, rhs);
}

Tensor operator>(const Tensor& lhs, const Tensor& rhs) {
  return gt(lhs, rhs);
}

Tensor operator>=(const Tensor& lhs, const Tensor& rhs) {
  return ge(lhs, rhs);
}

Tensor operator<(const Tensor& lhs, const Tensor& rhs) {
  return lt(lhs, rhs);
}

Tensor operator<=(const Tensor& lhs, const Tensor& rhs) {
  return le(lhs, rhs);
}

Tensor select(const Tensor& tensor, int axis, int index,
              bool keep_dim /* = false*/) {
  if (axis >= tensor.dim()) {
    throw ExceptionInvalidArgument("dim is invalid");
  }
  TensorShape shape_first_part;
  TensorShape shape_second_part;

  for (int i = 0; i < axis; i++) {
    shape_first_part.push_back(tensor.shape()[i]);
  }
  for (int i = axis + 1; i < tensor.dim(); i++) {
    shape_second_part.push_back(tensor.shape()[i]);
  }

  TensorShape result_shape(shape_first_part);
  if (keep_dim) {
    result_shape.push_back(1);
  }

  result_shape.insert(result_shape.end(), shape_second_part.begin(),
                      shape_second_part.end());

  Tensor result(result_shape);

  TensorIndices result_indices(result_shape.size(), 0);

  TensorIndices indices_first_part(axis, 0);
  TensorIndices indices_second_part(tensor.dim() - 1 - axis, 0);

  do {

    TensorIndices new_indices_first_part = indices_first_part;
    new_indices_first_part.push_back(index);

    do {
      TensorIndices reading_indices = TensorHelper::merge_indices(
          new_indices_first_part, indices_second_part);

      result.at(result_indices) = tensor.at(reading_indices);
      TensorHelper::increment_indices(result_indices, result_shape);

    } while (TensorHelper::increment_indices(indices_second_part,
                                             shape_second_part));
  } while (
      TensorHelper::increment_indices(indices_first_part, shape_first_part));

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  return result;
}

Tensor sum(const Tensor& tensor) {
  float sum_v = 0;
  const float* data = tensor.raw_data();
  for (int i = 0; i < tensor.data_size(); i++) {
    sum_v += *(data + i);
  }

  Tensor result(sum_v);
  UPDATE_BACKWARD_GRAPH(result, SumBackward, tensor);

  return result;
}

Tensor sum(const Tensor& tensor, int axis, bool keep_dim /* = false*/) {
  if (axis >= tensor.dim()) {
    throw ExceptionInvalidArgument("dim is invalid");
  }

  // need to add in-place operation to improve performance
  Tensor result = select(tensor, axis, 0, keep_dim);
  for (int i = 1; i < tensor.shape()[axis]; i++) {
    Tensor t = select(tensor, axis, i, keep_dim);
    result = result + t;
  }

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET(sum1, tensor);

  return result;
}

Tensor sum(const Tensor& tensor, const std::vector<int>& dims,
           bool keep_dim /* = false*/) {

  // Axes in dims should be in ascending order
  for (int i = 1; i < dims.size(); i++) {
    if (dims[i] <= dims[i - 1]) {
      throw ExceptionInvalidArgument(
          "axes in dims should be in ascending order");
    }
  }

  Tensor result(tensor);
  for (int i : dims) {
    result = result.sum(i, true);
  }

  if (!keep_dim) {
    // Performance can be improved here.
    for (int i = dims.size() - 1; i >= 0; i--) {
      result.shape().erase(result.shape().begin() + i);
      result.strides().erase(result.strides().begin() + i);
    }
  }

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET(sum2, tensor);

  return result;
}

Tensor mean(const Tensor& tensor) {
  // We don't need to update graph here, since it makes use of existing operations
  return sum(tensor) / tensor.data_size();
}

Tensor mean(const Tensor& tensor, int dim, bool keep_dim) {
  // We don't need to update graph here, since it makes use of existing operations
  int count = tensor.shape()[dim];
  return sum(tensor, dim, keep_dim) / count;
}

Tensor mean(const Tensor& tensor, const std::vector<int>& dims,
            bool keep_dim) {
  // We don't need to update graph here, since it makes use of existing operations
  int count = 1;
  for (auto dim : dims) {
    count *= tensor.shape()[dim];
  }

  return sum(tensor, dims, keep_dim) / count;
}

Tensor transpose(const Tensor& tensor) {
  if (tensor.dim() != 2) {
    throw ExceptionTensorShapeIncompatible();
  }

  return transpose(tensor, 0, 1);
}

Tensor transpose(const Tensor& tensor, int dim1, int dim2) {
  if (!(dim1 < tensor.dim() && dim2 < tensor.dim())) {
    throw ExceptionTensorShapeIncompatible();
  }

  // We will modify the result's meta info, so we need to make a meta copy
  // and leave the original tensor's meta info unchanged.
  Tensor result(tensor.meta_copy());
  std::swap(result.shape()[dim1], result.shape()[dim2]);
  std::swap(result.strides()[dim1], result.strides()[dim2]);

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET(transpose, tensor);

  return result;
}

Tensor sigmoid(const Tensor& tensor) {
  // We want the sigmoid's backward node to be single op node in the graph rather
  // than a series of arithmetic nodes. So we disable the grad mode before we do
  // arithmetic operations and add a single SigmoidBackward node at last.
  Tensor result = [&]() {
    autograd::GradModeGuard grad_guard(false);
    return div(Tensor(1), add(Tensor(1), exp(neg(tensor))));
  }();

  UPDATE_BACKWARD_GRAPH(result, SigmoidBackward, tensor);

  return result;
}

Tensor relu(const Tensor& tensor) {

  // We want the relu's backward node to be a single op node in the graph rather
  // than a series of arithmetic nodes. So we disable the grad mode before we do
  // arithmetic operations and add a single ReluBackward node at last.

  // Tensor result = (input + abs(input)) / 2;
  // return result;
  Tensor result = [&]() {
    autograd::GradModeGuard grad_guard(false);
    return div(add(tensor, abs(tensor)), Tensor(2));
  }();

  UPDATE_BACKWARD_GRAPH(result, ReluBackward, tensor);

  return result;
}

// Loss functions
Tensor smooth_l1_loss(const Tensor& input, const Tensor& target,
                      ReductionType rt, float beta) {

  Tensor result = [&]() {
    autograd::GradModeGuard guard(false);

    Tensor abs_diff = abs(input - target);

    Tensor r = where(abs_diff < beta, 0.5 * (abs_diff ^ 2) / beta,
                          abs_diff - 0.5 * beta);

    if (rt == ReductionType::Mean) {
      r = r.mean();
    } else if (rt == ReductionType::Sum) {
      r = r.sum();
    } else if (rt == ReductionType::None){
      // Nothing to do
    } else {
      throw ExceptionInvalidArgument("Unrecognized ReductionType");
    }

    return r;
  }();

  UPDATE_BACKWARD_GRAPH_2(result, SmoothL1LossBackward, rt, beta, input, target);

  return result;
}

// Tensor mse_loss() {}

}  // namespace toytorch
