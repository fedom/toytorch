#include <gtest/gtest.h>
#include "nn/tensor/tensor_creator.h"
#include "nn/exceptions/exceptions.h"

using namespace toytorch;

TEST(TensorCreatorTest, arange) {
  EXPECT_TRUE(arange(0, 11, 3) == Tensor({4}, {0, 3, 6, 9}));
  EXPECT_TRUE(arange(0, 11, 2) == Tensor({6}, { 0,  2,  4,  6,  8, 10}));
  EXPECT_TRUE(arange(1, 12, 3) == Tensor({4}, {1,  4,  7, 10}));
}