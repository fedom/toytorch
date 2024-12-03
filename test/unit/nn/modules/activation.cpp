#include "nn/modules/activation.h"
#include "nn/tensor/tensor.h"
#include <gtest/gtest.h>

using namespace toytorch;

TEST(ActivationTest, Sigmoid) {
  Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});

  nn::Sigmoid act;

  Tensor r = act.forward(t);
  // r.print();
  Tensor expect_result({2, 3}, {0.73106, 0.88080, 0.95257, 0.98201, 0.99331, 0.99753});

  EXPECT_TRUE(r.strict_allclose(expect_result));
}
