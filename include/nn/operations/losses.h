#ifndef TOYTORCH_NN_OPERATIONS_LOSSES_H__
#define TOYTORCH_NN_OPERATIONS_LOSSES_H__
#include "nn/tensor/tensor.h"
#include "nn/operations/common_types.h"

namespace toytorch {

// Loss functions
Tensor smooth_l1_loss(const Tensor& input, const Tensor& target,
                      ReductionType rt = ReductionType::Mean, float beta = 1.0);
Tensor mse_loss(const Tensor& input, const Tensor& target);

} // namespace toytorch

#endif // TOYTORCH_NN_OPERATIONS_LOSSES_H__