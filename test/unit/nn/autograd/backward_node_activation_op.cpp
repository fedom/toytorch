#include <gtest/gtest.h>
#include "nn/autograd/autograd.h"
#include "nn/autograd/backward_node_binary_op.h"
#include "nn/modules/activation.h"
#include "nn/tensor/tensor.h"
#include "nn/tensor/tensor_creator.h"
#include "nn/operations/tensor_operations.h"

using namespace toytorch;

TEST(AugogradTest, SigmoidBackward) {
  Sigmoid sig;
  Tensor t1({2, 3}, {1, 2, 3, 4, 5, 6}, true);

  Tensor t2 = sig.forward(t1);

  Tensor result = t2.sum() * 2;

  result.backward();

  // t1.grad()->print();

  EXPECT_TRUE(
      t1.grad()->strict_allclose(
          Tensor({2, 3}, {0.3932, 0.2100, 0.0904, 0.0353, 0.0133, 0.0049}),
          1e-2, 1e-8));
}

TEST(AugogradTest, ReluBackward) {
  Relu act;
  Tensor t1({2, 3}, {1, -2, 3, -4, 5, -6}, true);

  Tensor t2 = act.forward(t1);

  Tensor result = t2.sum() * 2;

  result.backward();

  // t1.grad()->print();

  EXPECT_TRUE(*t1.grad() == Tensor({2, 3}, {2, 0, 2, 0, 2, 0}));
}
