#ifndef TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_DROPOUT_OP_H__
#define TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_DROPOUT_OP_H__
#include "nn/autograd/node.h"

namespace toytorch::autograd {

class DropoutBackward : public UnaryNode {
 public:
  DEFINE_NODE_NAME_AND_ID(DropoutBackward)

  DropoutBackward(float scale, bool train, const Tensor& mask)
      : scale_(scale), train_(train), mask_(mask) {}

  Tensor calculate_grad(Tensor grad, Tensor input) override;

 private:
  Tensor mask_;
  float scale_;
  bool train_;
};

class Dropout2dBackward : public UnaryNode {
 public:
  DEFINE_NODE_NAME_AND_ID(Dropout2dBackward)

  Dropout2dBackward(bool train, const Tensor& mask)
      : train_(train), mask_(mask) {}

  Tensor calculate_grad(Tensor grad, Tensor input) override;

 private:
  Tensor mask_;
  bool train_;
};

}  // namespace toytorch::autograd

#endif  // TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_DROPOUT_OP_H__