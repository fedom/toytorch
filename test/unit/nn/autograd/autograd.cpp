#include "nn/autograd/autograd.h"
#include "nn/tensor/tensor.h"
#include "nn/tensor/tensor_operations.h"
#include "nn/tensor/tensor_creator.h"
#include "nn/debug/debug_utils.h"
#include <gtest/gtest.h>

using namespace toytorch;

TEST(AugogradTest, BackwardMultiPath) {
  Tensor t({2, 3}, {1, 1, 1, 1, 1, 1}, true);

  ASSERT_TRUE(t.requires_grad());

  Tensor t1 = pow(t, 2);                    // (2, 3)
  Tensor t2 = matmul(t, ones({3, 2}));      // (2, 2)
  Tensor t3 = matmul(t1, ones({3, 2}) * 3);  // (2, 2)

  Tensor t4 = t2 * t3;                      // (2, 2)

  Tensor result = t4.sum();

  ASSERT_TRUE(result.requires_grad());

  // std::cout << debug::print_backward_graph(result) << std::endl;

  result.backward();

  Tensor expect_result({2, 3}, {54, 54, 54, 54, 54, 54});

  EXPECT_TRUE(*t.grad_info()->grad == expect_result);
  // EXPECT_TRUE(result.strict_allclose(expect_result));
}
