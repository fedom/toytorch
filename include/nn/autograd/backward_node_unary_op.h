#ifndef TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_UNARY_OP_H__
#define TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_UNARY_OP_H__
#include "nn/autograd/node.h"

namespace toytorch::autograd {

class ExpBackward : public UnaryNode {
 public:
  DEFINE_NODE_NAME_AND_ID(ExpBackward)

  Tensor calculate_grad(Tensor grad, Tensor input) override;
};

class NegBackward : public UnaryNode {
 public:
  DEFINE_NODE_NAME_AND_ID(NegBackward)

  Tensor calculate_grad(Tensor grad, Tensor input) override;
};

class AbsBackward : public UnaryNode {
 public:
  DEFINE_NODE_NAME_AND_ID(AbsBackward)

  Tensor calculate_grad(Tensor grad, Tensor input) override;
};

class BernoulliBackward : public UnaryNode {
 public:
  DEFINE_NODE_NAME_AND_ID(BernoulliBackward)

  Tensor calculate_grad(Tensor grad, Tensor input) override;
};

class SumBackward : public UnaryNode {
 public:
  DEFINE_NODE_NAME_AND_ID(SumBackward)

  Tensor calculate_grad(Tensor grad, Tensor input) override;
};

class ViewBackward : public UnaryNode {
 public:
  DEFINE_NODE_NAME_AND_ID(ViewBackward)

  Tensor calculate_grad(Tensor grad, Tensor input) override;
};

class Pad2dBackward : public UnaryNode {
 public:
  DEFINE_NODE_NAME_AND_ID(Pad2dBackward)

  Pad2dBackward(int top, int bottom, int left, int right)
      : top_(top), bottom_(bottom), left_(left), right_(right) {}

  Tensor calculate_grad(Tensor grad, Tensor input) override;

 private:
  int top_;
  int bottom_;
  int left_;
  int right_;
};

class Pad1dBackward : public UnaryNode {
 public:
  DEFINE_NODE_NAME_AND_ID(Pad1dBackward)

  Pad1dBackward(int left, int right) : left_(left), right_(right) {}

  Tensor calculate_grad(Tensor grad, Tensor input) override;

 private:
  int left_;
  int right_;
};

}  // namespace toytorch::autograd

#endif  // TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_UNARY_OP_H__