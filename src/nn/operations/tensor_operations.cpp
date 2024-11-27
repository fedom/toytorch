#include "nn/autograd/autograd.h"
#include "nn/autograd/backward_node_activation_op.h"
#include "nn/autograd/backward_node_binary_op.h"
#include "nn/autograd/backward_node_leaf_op.h"
#include "nn/autograd/backward_node_unary_op.h"
#include "nn/operations/tensor_helper.h"
#include "nn/operations/tensor_operations.h"
#include "nn/exceptions/exceptions.h"

namespace toytorch {

using autograd::Edge;
using autograd::Node;

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

// https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold
// unfold doesn't require tensor is contiguous() but the result is an uncontiguous tensor 
Tensor unfold(const Tensor& tensor, int dim, int size, int step) {
  Tensor result(tensor.meta_copy());

  if (dim > tensor.dim() - 1) {
    throw ExceptionInvalidArgument("unfold dim exceed tensor dim range");
  }

  // the new shape_[dim] is calculated similarly to conv2d's height and width calculation
  int cur_dim_len = tensor.shape()[dim];
  int cur_dim_stride = tensor.strides()[dim];

  int cur_dim_new_len = ((cur_dim_len - (size - 1) - 1) / step) + 1;
  int cur_dim_new_stride = cur_dim_stride * step;

  int extended_dim_len = size;
  int extended_dim_stride = cur_dim_stride;

  result.shape()[dim] = cur_dim_new_len;
  result.strides()[dim] = cur_dim_new_stride;
  result.shape().push_back(extended_dim_len);;
  result.strides().push_back(extended_dim_stride);

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET(unfold, tensor);

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

  for (int i = 0; i < dims.size(); i++) {
    result = result.sum(dims[i] - (keep_dim ? 0 : i), keep_dim);
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

}  // namespace toytorch
