#include "nn/autograd/autograd.h"
#include "nn/tensor/tensor.h"
#include "nn/tensor/tensor_operations.h"
#include <gtest/gtest.h>

using namespace toytorch;

TEST(AugogradTest, backward) {
  Tensor t({2, 3}, {1, 1, 1, 1, 1, 1}, true);

  ASSERT_TRUE(t.requires_grad());

  Tensor result = pow(t, Tensor(2));

  ASSERT_TRUE(result.requires_grad());

  result = result.sum();

  ASSERT_TRUE(result.requires_grad());

  result.backward();

  Tensor expect_result({2, 3}, {2, 2, 2, 2, 2, 2});

  EXPECT_TRUE(*t.grad_info()->grad == expect_result);
  // EXPECT_TRUE(result.strict_allclose(expect_result));
}
