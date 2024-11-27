#include "nn/operations/losses.h"
#include "nn/tensor/tensor.h"
#include <gtest/gtest.h>

using namespace toytorch;

TEST(TensorOperationsTest, SmoothL1LossNone) {
  Tensor t1 = Tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  Tensor t2 = Tensor({2, 2, 2}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5});

  Tensor loss = smooth_l1_loss(t1, t2, ReductionType::None);

  // loss.print();
  EXPECT_TRUE(loss == Tensor({2, 2, 2}, {0.1250, 0.1250, 0.1250, 0.1250, 0.1250,
                                         0.1250, 0.1250, 0.1250}));
}

TEST(TensorOperationsTest, SmoothL1LossSum) {
  Tensor t1 = Tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  Tensor t2 = Tensor({2, 2, 2}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5});

  Tensor loss = smooth_l1_loss(t1, t2, ReductionType::Sum);

  // loss.print();
  EXPECT_TRUE(loss == Tensor(1));
}

TEST(TensorOperationsTest, SmoothL1LossMean) {
  Tensor t1 = Tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  Tensor t2 = Tensor({2, 2, 2}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5});

  Tensor loss = smooth_l1_loss(t1, t2, ReductionType::Mean);

  // loss.print();
  EXPECT_TRUE(loss == Tensor(0.125));
}
