#include "nn/operations/losses.h"
#include "nn/operations/tensor_operations.h"
#include "nn/autograd/autograd.h"
#include "nn/autograd/backward_node_activation_op.h"
#include "nn/autograd/backward_node_loss_op.h"
#include "nn/exceptions/exceptions.h"

namespace toytorch {


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

}