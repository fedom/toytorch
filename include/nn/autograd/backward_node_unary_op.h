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

} // namespace toytorch::autograd

#endif // TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_UNARY_OP_H__