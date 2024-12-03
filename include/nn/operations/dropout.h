#ifndef TOYTORCH_NN_OPERATIONS_DROPOUT_H__
#define TOYTORCH_NN_OPERATIONS_DROPOUT_H__
#include <tuple>
#include "nn/tensor/tensor.h"

namespace toytorch {

/**
 * @brief Refer to pytorch's dropout.
 * 
 * @param input 
 * @param p Probability to drop the value.
 * @param train True for train mode when dropout is activated. False for eval mode when dropout will deactivated.
 * @return Tensor 
 */
Tensor dropout(const Tensor& input, float p, bool train);
Tensor dropout2d(const Tensor& input, float p, bool train);

}  // namespace toytorch

#endif  // TOYTORCH_NN_OPERATIONS_DROPOUT_H__