#include "toytorch.h"
#include "nn/utils/print_utils.h"
#include <gtest/gtest.h>

using namespace toytorch;

// Also see unit/nn/modules/conv2d.cpp
TEST(ConvolutionTest, conv2d) {

  Tensor input = ones({5, 3, 8, 10});
  Tensor kernel = ones({20, 3, 2, 3});

  Tensor result = conv2d(input, kernel, {1, 1});

  // std::cout << result.shape() << std::endl;
  EXPECT_TRUE(result.shape() == TensorShape({5, 20, 7, 8}));

  // result.print();
  EXPECT_TRUE(result == ones({5, 20, 7, 8}) * 18);
}