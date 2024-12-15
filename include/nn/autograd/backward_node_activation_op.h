
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

class SoftmaxBackward : public UnaryNode {
 public:
  SoftmaxBackward(int dim, const Tensor &y) : dim_(dim), y_(y) {}
  Tensor calculate_grad(Tensor grad, Tensor input) override;

  DEFINE_NODE_NAME_AND_ID(SoftmaxBackward)

private:
  int dim_;
  Tensor y_;
};

class LogSoftmaxBackward : public UnaryNode {
 public:
  LogSoftmaxBackward(int dim) : dim_(dim) {}
  Tensor calculate_grad(Tensor grad, Tensor input) override;

  DEFINE_NODE_NAME_AND_ID(LogSoftmaxBackward)

private:
  int dim_;
};

}  // namespace toytorch::autograd

#endif  // TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_ACTIVATION_OP_H__
