#ifndef TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_CONVOLUTION_OP_H__
#define TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_CONVOLUTION_OP_H__
#include "nn/autograd/node.h"
#include "nn/operations/tensor_operations.h"

namespace toytorch::autograd {

class Conv2dBackward : public BinaryNode {
 public:
  Conv2dBackward(int hstride, int wstride)
      : hstride_(hstride),
        wstride_(wstride) {}

  Tensor calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;
  Tensor calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;

  DEFINE_NODE_NAME_AND_ID(Conv2dBackward)

 private:
  int hstride_;
  int wstride_;
};

class Conv1dBackward : public BinaryNode {
 public:
  Conv1dBackward(int stride) : stride_(stride) {}

  Tensor calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;
  Tensor calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) override;

  DEFINE_NODE_NAME_AND_ID(Conv1dBackward)

 private:
  int stride_;
};

}  // namespace toytorch::autograd

#endif  // TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_CONVOLUTION_OP_H__