#include "tensor_helper.h"
#include <cmath>
#include "nn/exceptions/exceptions.h"

namespace toytorch {

float greater_than(float a, float b) {
  return a > b ? 1 : 0;
}

float less_than(float a, float b) {
  return a < b ? 1 : 0;
}

float greater_eq(float a, float b) {
  return a >= b ? 1 : 0;
}

float less_eq(float a, float b) {
  return a <= b ? 1 : 0;
}

float choose_on_condition(float condition, float a, float b) {
  return condition > 0 ? a : b;
}

float get_sign(float a) {
  if (a > 0)
    return 1;
  if (a < 0)
    return -1;
  return 0;
}

std::function<float(float, float)> TensorHelper::binary_operations[] = {
    std::plus<float>(),        // EWBOP_ADD
    std::minus<float>(),       // EWBOP_SUB
    std::multiplies<float>(),  // EWBOP_MUL
    std::divides<float>(),     // EWBOP_DIV
    std::powf,                 // EWBOP_POW
    greater_than,              // EWBOP_GT
    less_than,                 // EWBOP_LT
    greater_eq,                // EWBOP_GE
    less_eq,                   // EWBOP_LE
};

std::function<float(float)>
    TensorHelper::unary_operations[EWUOP_NUM_OPERATIONS] = {
        std::expf,             // EWUOP_EXP
        std::negate<float>(),  // EWUOP_NEG
        std::fabsf,            // EWUOP_ABS
        get_sign,              // EWUOP_SIGN

};

std::function<float(float, float, float)>
    TensorHelper::ternary_operations[EWTOP_NUM_OPERATIONS] = {
        choose_on_condition,  // EWTOP_WHERE
};

bool TensorHelper::are_tensors_broadcastable(
    const std::vector<const Tensor*>& tensors, int num_skip_rightmost_dim) {
  if (!(num_skip_rightmost_dim == 0 || num_skip_rightmost_dim == 2)) {
    throw ExceptionInvalidArgument("skip dim must be 0 or 2");
  }

  int max_dim = 0;
  for (auto iter = tensors.begin(); iter != tensors.end(); iter++) {
    if ((*iter)->dim() < num_skip_rightmost_dim) {
      throw ExceptionInvalidArgument("tensor dim can't be less than skip dim");
    }
    max_dim = ((*iter)->dim() > max_dim ? (*iter)->dim() : max_dim);
  }

  std::vector<TensorShape> shapes;
  for (auto iter = tensors.begin(); iter != tensors.end(); iter++) {
    if ((*iter)->shape().size() < max_dim) {

      TensorShape shape = (*iter)->shape().expand_left_to_copy(max_dim);
      shapes.push_back(std::move(shape));
    } else {
      shapes.push_back((*iter)->shape());
    }
  }

  for (int i = max_dim - 1 - num_skip_rightmost_dim; i >= 0; i--) {
    int cur_dim = 0;
    for (int j = 0; j < shapes.size(); j++) {
      if (shapes[j][i] == 1) {
        continue;
      } else if (!cur_dim) {
        cur_dim = shapes[j][i];
      } else if (shapes[j][i] == cur_dim) {
        continue;
      } else {
        return false;
      }
    }
  }

  return true;
}

void TensorHelper::broadcast_tensors(const std::vector<Tensor*>& tensors,
                                     int num_skip_rightmost_dim) {

  // caller ensure are_tensors_broadcastable() are called and return true beforehand
  // assert(are_tensors_broadcastable(tensors, num_skip_rightmost_dim));

  if (!(num_skip_rightmost_dim == 0 || num_skip_rightmost_dim == 2)) {
    throw ExceptionInvalidArgument("skip dim must be 0 or 2");
  }

  int max_dim = 0;
  for (auto& tensor : tensors) {
    if (tensor->dim() < num_skip_rightmost_dim) {
      throw ExceptionInvalidArgument("tensor dim can't be less than skip dim");
    }
    max_dim = (tensor->dim() > max_dim ? tensor->dim() : max_dim);
  }

  for (auto& tensor : tensors) {
    tensor->shape().expand_left_to(max_dim, 1);
    tensor->strides().expand_left_to(max_dim, 0);
  }

  for (int i = max_dim - 1 - num_skip_rightmost_dim; i >= 0; i--) {
    int cur_dim = 1;
    for (int j = 0; j < tensors.size(); j++) {
      if (tensors[j]->shape()[i] != 1) {
        cur_dim = tensors[j]->shape()[i];
        break;
      }
    }
    for (int j = 0; j < tensors.size(); j++) {
      assert(tensors[j]->shape()[i] == 1 || tensors[j]->shape()[i] == cur_dim);
      if (tensors[j]->shape()[i] == 1 && cur_dim != 1) {
        tensors[j]->shape()[i] = cur_dim;
        tensors[j]->strides()[i] = 0;
      }
    }
  }
}

bool TensorHelper::are_tensors_broadcastable(const Tensor& tensor_a,
                                             const Tensor& tensor_b,
                                             int num_skip_rightmost_dim) {

  return are_tensors_broadcastable({&tensor_a, &tensor_b},
                                   num_skip_rightmost_dim);
}

bool TensorHelper::are_tensors_broadcastable(const Tensor& tensor_a,
                                             const Tensor& tensor_b,
                                             const Tensor& tensor_c,
                                             int num_skip_rightmost_dim) {

  return are_tensors_broadcastable({&tensor_a, &tensor_b, &tensor_c},
                                   num_skip_rightmost_dim);
}

void TensorHelper::broadcast_tensors(Tensor& tensor_a, Tensor& tensor_b,
                                     int num_skip_rightmost_dim) {
  broadcast_tensors({&tensor_a, &tensor_b}, num_skip_rightmost_dim);
}

void TensorHelper::broadcast_tensors(Tensor& tensor_a, Tensor& tensor_b,
                                     Tensor& tensor_c,
                                     int num_skip_rightmost_dim) {
  broadcast_tensors({&tensor_a, &tensor_b, &tensor_c}, num_skip_rightmost_dim);
}

float TensorHelper::apply_unary_op(ElementwiseUnaryOperation op, float a) {
  return unary_operations[op](a);
}

Tensor TensorHelper::elementwise_unary_op(const Tensor& tensor,
                                          ElementwiseUnaryOperation op) {
  Tensor result = tensor.deep_copy();
  for (int i = 0; i < result.numel(); i++) {
    result[i] = apply_unary_op(op, result[i]);
  }
  return result;
}

float TensorHelper::apply_binary_op(ElementwiseBinaryOperation op, float a,
                                    float b) {
  return binary_operations[op](a, b);
}

Tensor TensorHelper::elementwise_binary_op_scalar(
    const Tensor& tensor_a, const Tensor& tensor_b,
    ElementwiseBinaryOperation op) {
  assert(tensor_a.is_scalar() || tensor_b.is_scalar());

  Tensor result;
  if (tensor_a.is_scalar()) {
    result = tensor_b.deep_copy();
    float scalar_val = tensor_a[0];
    for (int i = 0; i < result.numel(); i++) {
      result[i] = apply_binary_op(op, scalar_val, result[i]);
    }
  } else {
    result = tensor_a.deep_copy();
    float scalar_val = tensor_b[0];
    for (int i = 0; i < result.numel(); i++) {
      result[i] = apply_binary_op(op, result[i], scalar_val);
    }
  }

  return result;
}

Tensor TensorHelper::elementwise_binary_op(const Tensor& tensor_a,
                                           const Tensor& tensor_b,
                                           ElementwiseBinaryOperation op) {

  if (tensor_a.is_scalar() || tensor_b.is_scalar()) {
    return TensorHelper::elementwise_binary_op_scalar(tensor_a, tensor_b, op);
  }

  assert(tensor_a.dim() > 0 && tensor_b.dim() > 0);

  if (!are_tensors_broadcastable(tensor_a, tensor_b, 0)) {
    throw ExceptionTensorShapeIncompatible();
  }

  // Note: These two copied tensors are sharing the raw data with original ones.
  //       So we should regard them as read-only.
  Tensor broadcasted_tensor_a(tensor_a.meta_copy());
  Tensor broadcasted_tensor_b(tensor_b.meta_copy());

  broadcast_tensors(broadcasted_tensor_a, broadcasted_tensor_b, 0);

  Tensor result(broadcasted_tensor_a.shape());

  TensorIndices result_indices = result.get_indices();
  TensorIndicesWalker walker(result.shape(), result_indices);
  do {
    result.at(result_indices) =
        apply_binary_op(op, broadcasted_tensor_a.at(result_indices),
                        broadcasted_tensor_b.at(result_indices));

  } while (walker.step());

  return result;
}

void TensorHelper::elementwise_binary_op_inplace(
    Tensor& tensor_a, const Tensor& tensor_b, ElementwiseBinaryOperation op) {

  if (!are_tensors_broadcastable(tensor_a, tensor_b, 0)) {
    throw ExceptionTensorShapeIncompatible();
  }

  // Note: These two copied tensors are sharing the raw data with original ones.
  //       So we should regard them as read-only.
  Tensor broadcasted_tensor_a(tensor_a.meta_copy());
  Tensor broadcasted_tensor_b(tensor_b.meta_copy());

  broadcast_tensors(broadcasted_tensor_a, broadcasted_tensor_b, 0);

  if (broadcasted_tensor_a.shape() != tensor_a.shape()) {
    throw ExceptionTensorShapeIncompatible(
        "First tensor's shape must match the output's shape in an inplace "
        "operation");
  }

  TensorIndices indices = tensor_a.get_indices();
  TensorIndicesWalker walker(tensor_a.shape(), indices);

  do {
    tensor_a.at(indices) =
        apply_binary_op(op, tensor_a.at(indices),
                        broadcasted_tensor_b.at(indices));

  } while (walker.step());
}

float TensorHelper::apply_ternary_op(ElementwiseTernaryOperation op, float a,
                                     float b, float c) {
  return ternary_operations[op](a, b, c);
}

Tensor TensorHelper::elementwise_ternary_op(const Tensor& tensor_a,
                                            const Tensor& tensor_b,
                                            const Tensor& tensor_c,
                                            ElementwiseTernaryOperation op) {

  if (!are_tensors_broadcastable(tensor_a, tensor_b, tensor_c, 0)) {
    throw ExceptionTensorShapeIncompatible();
  }

  // Note: These copied tensors are sharing the raw data with original ones.
  //       So we should regard them as read-only.
  Tensor broadcasted_tensor_a(tensor_a.meta_copy());
  Tensor broadcasted_tensor_b(tensor_b.meta_copy());
  Tensor broadcasted_tensor_c(tensor_c.meta_copy());

  broadcast_tensors(broadcasted_tensor_a, broadcasted_tensor_b,
                    broadcasted_tensor_c, 0);

  Tensor result(broadcasted_tensor_a.shape());
  TensorIndices result_indices = result.get_indices();
  TensorIndicesWalker walker(result.shape(), result_indices);
  do {
    result.at(result_indices) =
        apply_ternary_op(op, broadcasted_tensor_a.at(result_indices),
                         broadcasted_tensor_b.at(result_indices),
                         broadcasted_tensor_c.at(result_indices));

  } while (walker.step());

  return result;
}

}  //namespace toytorch