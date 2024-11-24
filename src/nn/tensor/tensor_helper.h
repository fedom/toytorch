#ifndef SRC_NN_TENSOR_TENSOR_HELPER_H__
#define SRC_NN_TENSOR_TENSOR_HELPER_H__
#include <initializer_list>
#include "nn/tensor/tensor.h"

namespace toytorch {

class TensorHelper {
 public:
  /**
   * @brief Check if two tensors fullfil the constraints for broadcast.
   * 
   * @param tensor_a 
   * @param tensor_b 
   * @param num_skip_rightmost_dim This is the number of rightmost dim that we need to skip for 
   * broadcast constraint. For element-wise operation, where we need to apply broadcast to the whole
   * shape of the tensor, this would be 0. For operations like matmul() between two matrix (2d or more),
   * we would skip the last two dimensions, since they should be imposed with constraints for matrix
   * multiplication (for matrix of shape (m,n) and (p, q), n == p). In that case, this number should be 2.
   * 
   * 
   * @return true 
   * @return false 
   */
  static bool are_tensors_broadcastable(const Tensor& tensor_a,
                                        const Tensor& tensor_b,
                                        int num_skip_rightmost_dim = 0);

  static bool are_tensors_broadcastable(const Tensor& tensor_a,
                                        const Tensor& tensor_b,
                                        const Tensor& tensor_c,
                                        int num_skip_rightmost_dim = 0);
  /**
   * @brief Broadcast the passed in tensor. Caller should guarantee they are broadcastable. Call
   * is_broadcastable before this.
   * 
   * @param tensor_a 
   * @param tensor_b 
   * @param num_skip_rightmost_dim See is_tensor_broadcastable().
   */
  static void broadcast_tensors(Tensor& tensor_a, Tensor& tensor_b,
                                int num_skip_rightmost_dim = 0);
  static void broadcast_tensors(Tensor& tensor_a, Tensor& tensor_b,
                                Tensor& tensor_c,
                                int num_skip_rightmost_dim = 0);

  static TensorIndices merge_indices(const TensorIndices& indices1,
                                     const TensorIndices& indices2) {
    TensorIndices result_indices(indices1);
    result_indices.insert(result_indices.end(), indices2.begin(),
                          indices2.end());
    return result_indices;
  }

  /**
   * @brief 
   * 
   * @param indices 
   * @param shape 
   * @return true if indices hasn't reach end
   * @return false if indices reaches end
   */
  static bool increment_indices(TensorIndices& indices,
                                const TensorShape& shape) {
    assert(indices.size() == shape.size());

    for (int i = shape.size() - 1; i >= 0; i--) {
      if (++indices[i] < shape[i]) {
        return true;
      }
      indices[i] = 0;
    }

    return false;
  }

  enum ElementwiseUnaryOperation {
    EWUOP_EXP,
    EWUOP_NEG,
    EWUOP_ABS,
    EWUOP_SIGN,
    EWUOP_NUM_OPERATIONS
  };

  static std::function<float(float)> unary_operations[EWUOP_NUM_OPERATIONS];

  static float apply_unary_op(ElementwiseUnaryOperation op, float a);

  static Tensor elementwise_unary_op(const Tensor& tensor,
                                     ElementwiseUnaryOperation op);

  enum ElementwiseBinaryOperation {
    EWBOP_ADD,
    EWBOP_SUB,
    EWBOP_MUL,
    EWBOP_DIV,
    EWBOP_POW,
    EWBOP_GT,
    EWBOP_LT,
    EWBOP_GE,
    EWBOP_LE,
    EWBOP_NUM_OPERATIONS,
  };

  static std::function<float(float, float)>
      binary_operations[EWBOP_NUM_OPERATIONS];

  static float apply_binary_op(ElementwiseBinaryOperation op, float a, float b);

  static Tensor elementwise_binary_op_scalar(const Tensor& tensor_a,
                                             const Tensor& tensor_b,
                                             ElementwiseBinaryOperation op);

  static Tensor elementwise_binary_op(const Tensor& tensor_a,
                                      const Tensor& tensor_b,
                                      ElementwiseBinaryOperation op);

  static void elementwise_binary_op_inplace(Tensor& tensor_a,
                                            const Tensor& tensor_b,
                                            ElementwiseBinaryOperation op);

  enum ElementwiseTernaryOperation {
    EWTOP_WHERE,
    EWTOP_NUM_OPERATIONS,
  };

  static std::function<float(float, float, float)>
      ternary_operations[EWTOP_NUM_OPERATIONS];

  static float apply_ternary_op(ElementwiseTernaryOperation op, float a,
                                float b, float c);

  static Tensor elementwise_ternary_op(const Tensor& tensor_a,
                                       const Tensor& tensor_b,
                                       const Tensor& tensor_c,
                                       ElementwiseTernaryOperation op);

 private:
  static bool are_tensors_broadcastable(
      const std::vector<const Tensor*>& tensors, int num_skip_rightmost_dim);

  static void broadcast_tensors(const std::vector<Tensor*>& tensors,
                                int num_skip_rightmost_dim);
};

}  // namespace toytorch

#endif  // SRC_NN_TENSOR_TENSOR_HELPER_H__