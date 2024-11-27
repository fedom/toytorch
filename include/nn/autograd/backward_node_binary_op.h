#ifndef TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_BINARY_OP_H__
#define TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_BINARY_OP_H__
#include "nn/autograd/node.h"
#include "nn/operations/tensor_operations.h"

namespace toytorch::autograd {

class MatmulBackward : public BinaryNode {
 public:
  Tensor calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;
  Tensor calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;

  DEFINE_NODE_NAME_AND_ID(MatmulBackward)
};

class AddBackward : public BinaryNode {
 public:
  Tensor calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;
  Tensor calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;

  DEFINE_NODE_NAME_AND_ID(AddBackward)

};

class SubBackward : public BinaryNode {
 public:
  Tensor calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;
  Tensor calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;

  DEFINE_NODE_NAME_AND_ID(SubBackward)

};

class MulBackward : public BinaryNode {
 public:
  Tensor calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;
  Tensor calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;

  DEFINE_NODE_NAME_AND_ID(MulBackward)

};

class DivBackward : public BinaryNode {
 public:
  Tensor calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;
  Tensor calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;

  DEFINE_NODE_NAME_AND_ID(DivBackward)

};

class PowBackward : public BinaryNode {
 public:
  Tensor calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;
  Tensor calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;

  DEFINE_NODE_NAME_AND_ID(PowBackward)

};

class WhereBackward : public BinaryNode {
 public:
  WhereBackward(const Tensor& condition) : condition_(condition) {}

  Tensor calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;
  Tensor calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;

  DEFINE_NODE_NAME_AND_ID(WhereBackward)

private:
  Tensor condition_;
};

}  // namespace toytorch::autograd

#endif  // TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_BINARY_OP_H__