#include <gtest/gtest.h>
#include "nn/autograd/backward_node_binary_op.h"
#include "nn/autograd/autograd.h"
#include "nn/tensor/tensor.h"
#include "nn/tensor/tensor_creator.h"
#include "nn/operations/tensor_operations.h"
#include "nn/operations/matrix.h"
#include "nn/debug/debug_utils.h"

using namespace toytorch;

TEST(AugogradUnaryTest, View) {

  Tensor t1 = ones({2,3,4,5}, true);

  Tensor t2 = t1.view({6, 20});

  Tensor t3 = ones({20, 3}) * 4;

  Tensor t4 = matmul(t2, t3);

  Tensor t5 = t4.mean();

  t5.backward();

  EXPECT_TRUE(t1.grad()->strict_allclose(t1 * 0.6667, 1e-6, 1e-4));
}