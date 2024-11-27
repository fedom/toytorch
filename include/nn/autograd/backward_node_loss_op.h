#ifndef TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_LOSS_OP_H__
#define TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_LOSS_OP_H__
#include "nn/autograd/node.h"
#include "nn/operations/common_types.h"
#include "nn/operations/tensor_operations.h"

namespace toytorch::autograd {

class SmoothL1LossBackward : public BinaryNode {
 public:
  SmoothL1LossBackward(const ReductionType rt, float beta) : rt_(rt), beta_(beta) {}

  Tensor calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;
  Tensor calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;

  DEFINE_NODE_NAME_AND_ID(SmoothL1LossBackward)

private:
  ReductionType rt_;
  float beta_;
};

} // namespace toytorch::autograd

#endif // TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_LOSS_OP_H__