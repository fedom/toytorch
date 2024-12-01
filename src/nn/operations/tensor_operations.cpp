#include "nn/operations/tensor_operations.h"
#include <iostream>
#include "nn/autograd/autograd.h"
#include "nn/autograd/backward_node_activation_op.h"
#include "nn/autograd/backward_node_binary_op.h"
#include "nn/autograd/backward_node_leaf_op.h"
#include "nn/autograd/backward_node_unary_op.h"
#include "nn/exceptions/exceptions.h"
#include "nn/operations/tensor_helper.h"
#include "nn/utils/print_utils.h"

namespace toytorch {

inline int normalize_dim(const Tensor& t, int dim) {
  return dim < 0 ? (t.dim() + dim) : dim;
}

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

Tensor sign(const Tensor& tensor) {
  Tensor result =
      TensorHelper::elementwise_unary_op(tensor, TensorHelper::EWUOP_SIGN);

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET("sign", tensor);

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

  result.strides().insert(dim, stride);
  result.shape().insert(dim, 1);

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET("unsqueeze", tensor);

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

  result.shape().remove(dim);
  result.strides().remove(dim);

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET("squeeze", tensor);

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
  result.shape().push_back(extended_dim_len);
  result.strides().push_back(extended_dim_stride);

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET("unfold", tensor);

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

  Tensor result(result_shape);

  TensorIndices result_indices = result.get_indices();
  TensorIndicesWalker result_walker(result_shape, result_indices);

  int next_start_index = 0;
  for (auto& tensor : tensors) {
    TensorIndices tensor_indices = tensor.get_indices();
    TensorIndicesWalker tensor_walker(tensor.shape(), tensor_indices);

    result_walker.narrow_to_index_range(dim, next_start_index, tensor.shape()[dim]);
    next_start_index += tensor.shape()[dim];
    do
    {
      result.at(result_indices) = tensor.at(tensor_indices);
      result_walker.step();

    } while (tensor_walker.step());
  }

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET_VEC("cat", tensors);

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

  TensorShape result_shape = tensor.shape();

  if (keep_dim) {
    result_shape[axis] = 1;
  } else {
    result_shape.remove(axis);
  }
  
  Tensor result(result_shape);

  TensorIndices read_indices = tensor.get_indices();
  TensorIndicesWalker read_walker(tensor.shape(), read_indices);
  read_walker.narrow_to_index(axis, index);

  int i = 0;
  do {
    result.at_raw(i++) = tensor.at(read_indices);
  } while (read_walker.step());

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  return result;
}

Tensor sum(const Tensor& tensor) {
  float sum_v = 0;
  const float* data = tensor.raw_data();
  for (int i = 0; i < tensor.numel(); i++) {
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
  BACKWARD_NOT_IMPLEMENTED_YET("sum1", tensor);

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
  BACKWARD_NOT_IMPLEMENTED_YET("sum2", tensor);

  return result;
}

Tensor mean(const Tensor& tensor) {
  // We don't need to update graph here, since it makes use of existing operations
  return sum(tensor) / tensor.numel();
}

Tensor mean(const Tensor& tensor, int dim, bool keep_dim) {
  // We don't need to update graph here, since it makes use of existing operations
  int count = tensor.shape()[dim];
  return sum(tensor, dim, keep_dim) / count;
}

Tensor mean(const Tensor& tensor, const std::vector<int>& dims, bool keep_dim) {
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

  dim1 = normalize_dim(tensor, dim1);
  dim2 = normalize_dim(tensor, dim2);

  if (!(dim1 >= 0 && dim1 < tensor.dim() && dim2 >= 0 && dim2 < tensor.dim())) {
    throw ExceptionInvalidArgument("transpose args dim out of range");
  }

  // We will modify the result's meta info, so we need to make a meta copy
  // and leave the original tensor's meta info unchanged.
  Tensor result(tensor.meta_copy());
  std::swap(result.shape()[dim1], result.shape()[dim2]);
  std::swap(result.strides()[dim1], result.strides()[dim2]);

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET("transpose", tensor);

  return result;
}

Tensor slice(const Tensor& tensor, int dim, int start, int end) {

  // we support negative dim index, -1 is the last
  if (dim < 0) {
    dim = tensor.dim() + dim;
  }

  if (dim < 0 || dim >= tensor.dim()) {
    throw ExceptionInvalidArgument("slice() arg dim out of range");
  }
  if (!(start >= 0 && end <= tensor.dim(dim) && start < end)) {
    throw ExceptionInvalidArgument("slice() args start & end are not valid");
  }

  Tensor result(tensor.meta_copy());

  result.shape()[dim] = end - start;
  int offset = result.strides()[dim] * start;
  result.set_offset(offset + result.offset());

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET("transpose", tensor);

  return result;
}

Tensor flip(const Tensor& input, const std::vector<int>& dims) {
  Tensor result = input.deep_copy();

  TensorIndices input_indices = input.get_indices();
  TensorIndices result_indices = result.get_indices();

  TensorIndicesWalker input_walker(input.shape(), input_indices);
  TensorIndicesWalker result_walker(result.shape(), result_indices);

  for (auto dim : dims) {
    dim = normalize_dim(input, dim);
    
    for (int i = 0; i < input.shape()[dim] / 2; i++) {
      result_walker.narrow_to_index(dim, i);
      do
      {
        float &a = result.at(result_indices);
        result_indices[dim] = result.shape()[dim] - 1 - i;
        float &b = result.at(result_indices);
        result_indices[dim] = i;
        std::swap(a, b);
      } while (result_walker.step());

      input_walker.reset();
      result_walker.reset();
    }
  }

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET("flip", input);

  return result;
}

}  // namespace toytorch
