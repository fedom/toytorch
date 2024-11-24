#include <gtest/gtest.h>
#include "nn/autograd/backward_node_binary_op.h"
#include "nn/autograd/autograd.h"
#include "nn/tensor/tensor.h"
#include "nn/tensor/tensor_creator.h"
#include "nn/tensor/tensor_operations.h"
#include "nn/debug/debug_utils.h"

using namespace toytorch;

TEST(AugogradTest, MatmulBackward) {
  Tensor a = ones({2, 4}, true);
  Tensor b = ones({4, 3}, true);

  Tensor c = matmul(a * 2, b * 3);
  Tensor d = c.sum();

  d.backward();

  // a.grad()->print();
  // b.grad()->print();

  EXPECT_TRUE(*a.grad() == Tensor({2, 4}, 18));
  EXPECT_TRUE(*b.grad() == Tensor({4, 3}, 12));
}

TEST(AugogradTest, AddBackward) {
  Tensor a = ones({2, 4}, true);
  Tensor b = ones({1, 4}, true);

  Tensor c = a + b;
  Tensor d = c.sum();

  d.backward();

  // std::cout << debug::print_backward_graph(d) << std::endl;

  EXPECT_TRUE(*a.grad() == Tensor({2, 4}, 1));
  EXPECT_TRUE(*b.grad() == Tensor({1, 4}, 2));
}

TEST(AugogradTest, SubBackward) {
  Tensor a = ones({2, 4}, true);
  Tensor b = ones({2, 4}, true);

  Tensor c = a - b;
  Tensor d = c.sum();

  d.backward();

  EXPECT_TRUE(*a.grad() == Tensor({2, 4}, 1));
  EXPECT_TRUE(*b.grad() == Tensor({2, 4}, -1));
}

TEST(AugogradTest, MulBackward) {
  Tensor a = Tensor({2, 3}, 2, true);
  Tensor b = Tensor({2, 3}, 3, true);

  Tensor c = a * b;
  Tensor d = c.sum();

  d.backward();

  EXPECT_TRUE(*a.grad() == Tensor({2, 3}, 3));
  EXPECT_TRUE(*b.grad() == Tensor({2, 3}, 2));
}

TEST(AugogradTest, DivBackward) {
  Tensor a = Tensor({2, 4}, 4, true);
  Tensor b = Tensor({2, 4}, 2, true);

  Tensor c = a / b;
  Tensor d = c.sum();

  d.backward();

  // a.grad()->print();
  // b.grad()->print();

  EXPECT_TRUE(*a.grad() == Tensor({2, 4}, 0.5));
  EXPECT_TRUE(*b.grad() == Tensor({2, 4}, -1));
}

TEST(AugogradTest, PowBackward) {
  Tensor a = Tensor({2, 3}, 2, true);
  Tensor b = Tensor({2, 3}, 3, false);

  Tensor c = pow(a, b);
  Tensor d = c.sum();

  d.backward();

  EXPECT_TRUE(*a.grad() == Tensor({2, 3}, 12));
  // EXPECT_TRUE(*b.grad() == Tensor({2, 3}, ));
}

TEST(AugogradTest, WhereBackward) {
  Tensor condition = Tensor({2, 3}, {1, 0, 0, 1, 1, 1}, true);
  Tensor a = Tensor({2, 3}, 2, true);
  Tensor b = Tensor({2, 3}, 3, true);

  Tensor c = where(condition, a, b);
  Tensor d = c.sum();

  d.backward();

  EXPECT_TRUE(*a.grad() == Tensor({2, 3}, {1, 0, 0, 1, 1, 1}));
  EXPECT_TRUE(*b.grad() == Tensor({2, 3}, {0, 1, 1, 0, 0, 0}));
  EXPECT_TRUE(*condition.grad() == Tensor(0));
  
  // EXPECT_TRUE(*b.grad() == Tensor({2, 3}, ));
}

TEST(AugogradTest, SmoothL1LossBackwardMean) {
  Tensor t1 = Tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, true);
  Tensor t2 = Tensor({2, 2, 2}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5}, true);

  Tensor loss = smooth_l1_loss(t1, t2, ReductionType::Mean);

  loss.backward();

  // t1.grad()->print();
  // t2.grad()->print();

  EXPECT_TRUE(*t1.grad() == Tensor({2,2,2}, {-0.0625, -0.0625,-0.0625, -0.0625,-0.0625, -0.0625,-0.0625, -0.0625}));
  EXPECT_TRUE(*t2.grad() == Tensor({2,2,2}, {0.0625, 0.0625,0.0625, 0.0625,0.0625, 0.0625,0.0625, 0.0625}));
}

TEST(AugogradTest, SmoothL1LossBackwardSum) {
  Tensor t1 = Tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, true);
  Tensor t2 = Tensor({2, 2, 2}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5}, true);

  Tensor loss = smooth_l1_loss(t1, t2, ReductionType::Sum);

  loss.backward();

  // t1.grad()->print();
  // t2.grad()->print();

  EXPECT_TRUE(*t1.grad() == Tensor({2,2,2}, {-0.5, -0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,}));
  EXPECT_TRUE(*t2.grad() == Tensor({2,2,2}, {0.5, 0.5,0.5,0.5,0.5,0.5,0.5,0.5,}));
}

TEST(AugogradTest, SmoothL1LossBackwardNone) {
  Tensor t1 = Tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, true);
  Tensor t2 = Tensor({2, 2, 2}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5}, true);

  Tensor loss = smooth_l1_loss(t1, t2, ReductionType::None);

  Tensor loss_sum = loss.sum();

  loss_sum.backward();

  // t1.grad()->print();
  // t2.grad()->print();

  EXPECT_TRUE(*t1.grad() == Tensor({2,2,2}, {-0.5, -0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,}));
  EXPECT_TRUE(*t2.grad() == Tensor({2,2,2}, {0.5, 0.5,0.5,0.5,0.5,0.5,0.5,0.5,}));
}