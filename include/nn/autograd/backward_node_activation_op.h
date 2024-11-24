
#ifndef TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_ACTIVATION_OP_H__
#define TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_ACTIVATION_OP_H__
#include "nn/autograd/node.h"

namespace toytorch::autograd {
class SigmoidBackward : public UnaryNode {
 public:
  Tensor calculate_grad(Tensor grad, Tensor input) override;

  DEFINE_NODE_NAME_AND_ID(SigmoidBackward)
};

class ReluBackward : public UnaryNode {
 public:
  Tensor calculate_grad(Tensor grad, Tensor input) override;

  DEFINE_NODE_NAME_AND_ID(ReluBackward)
};

}  // namespace toytorch::autograd

#endif  // TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_ACTIVATION_OP_H__
