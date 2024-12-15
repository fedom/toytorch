#include "nn/autograd/backward_node_activation_op.h"
#include <gtest/gtest.h>
#include "nn/autograd/autograd.h"
#include "nn/autograd/backward_node_binary_op.h"
#include "nn/debug/debug_utils.h"
#include "nn/modules/activation.h"
#include "nn/operations/activations.h"
#include "nn/operations/tensor_operations.h"
#include "nn/tensor/tensor.h"
#include "nn/tensor/tensor_creator.h"

using namespace toytorch;

TEST(AugogradTest, SigmoidBackward) {
  nn::Sigmoid sig;
  Tensor t1({2, 3}, {1, 2, 3, 4, 5, 6}, true);

  Tensor t2 = sig.forward(t1);

  Tensor result = t2.sum() * 2;

  result.backward();

  // t1.grad()->print();

  EXPECT_TRUE(t1.grad()->strict_allclose(
      Tensor({2, 3}, {0.3932, 0.2100, 0.0904, 0.0353, 0.0133, 0.0049}), 1e-2,
      1e-8));
}

TEST(AugogradTest, ReluBackward) {
  nn::Relu act;
  Tensor t1({2, 3}, {1, -2, 3, -4, 5, -6}, true);

  Tensor t2 = act.forward(t1);

  Tensor result = t2.sum() * 2;

  result.backward();

  // t1.grad()->print();

  EXPECT_TRUE(*t1.grad() == Tensor({2, 3}, {2, 0, 2, 0, 2, 0}));
}

TEST(AugogradTest, SoftmaxBackward) {

  Tensor a = Tensor({2, 3, 4}, {0.2412, 0.4876, 0.5618, 0.6176, 0.6800, 0.9256,
                                0.5233, 0.6787, 0.3179, 0.1484, 0.4914, 0.8708,
                                0.7588, 0.5124, 0.4382, 0.3824, 0.3200, 0.0744,
                                0.4767, 0.3213, 0.6821, 0.8516, 0.5086, 0.1292},
                    true);

  Tensor target = ones({2, 3, 4});

  Tensor b = softmax(a, 0);

  ASSERT_TRUE(b.strict_allclose(
      Tensor({2, 3, 4},
             {0.3734, 0.4938, 0.5309, 0.5585, 0.5890, 0.7008, 0.5116, 0.5884,
              0.4099, 0.3311, 0.4957, 0.6773, 0.6266, 0.5062, 0.4691, 0.4415,
              0.4110, 0.2992, 0.4884, 0.4116, 0.5901, 0.6689, 0.5043, 0.3227}),
      1e-6, 1e-4));

  // result.print();
  Tensor c = sum(target * toytorch::log(b));

  // std::cout << debug::print_backward_graph(c);
  c.backward();

  // c.print();
  // a.grad()->print();
  EXPECT_TRUE(a.grad()->strict_allclose(
      Tensor({2, 3, 4}, {0.2532,  0.0124,  -0.0617, -0.1171, -0.1781, -0.4016,
                         -0.0233, -0.1768, 0.1801,  0.3378,  0.0086,  -0.3547,
                         -0.2532, -0.0124, 0.0617,  0.1171,  0.1781,  0.4016,
                         0.0233,  0.1768,  -0.1801, -0.3378, -0.0086, 0.3547}),
      1e-6, 1e-4));
}

TEST(AugogradTest, SoftmaxBackwardDim1) {

  Tensor a = Tensor({2, 3, 4}, {0.2412, 0.4876, 0.5618, 0.6176, 0.6800, 0.9256,
                                0.5233, 0.6787, 0.3179, 0.1484, 0.4914, 0.8708,
                                0.7588, 0.5124, 0.4382, 0.3824, 0.3200, 0.0744,
                                0.4767, 0.3213, 0.6821, 0.8516, 0.5086, 0.1292},
                    true);

  Tensor target = ones({2, 3, 4});

  Tensor b = softmax(a, 1);

  ASSERT_TRUE(b.strict_allclose(
      Tensor({2, 3, 4},
             {0.2754, 0.3066, 0.3455, 0.2984, 0.4272, 0.4751, 0.3325, 0.3172,
              0.2974, 0.2184, 0.3220, 0.3844, 0.3890, 0.3280, 0.3213, 0.3680,
              0.2508, 0.2116, 0.3339, 0.3462, 0.3602, 0.4604, 0.3448, 0.2857}),
      1e-6, 1e-4));

  // result.print();
  Tensor c = sum(target * toytorch::log(b));

  // std::cout << debug::print_backward_graph(c);
  c.backward();

  // c.print();
  // a.grad()->print();
  EXPECT_TRUE(a.grad()->strict_allclose(
      Tensor({2, 3, 4}, {0.1737,  0.0803,  -0.0365, 0.1048,  -0.2815, -0.4252,
                         0.0026,  0.0484,  0.1078,  0.3449,  0.0339,  -0.1532,
                         -0.1669, 0.0161,  0.0360,  -0.1041, 0.2476,  0.3651,
                         -0.0018, -0.0387, -0.0807, -0.3812, -0.0343, 0.1428}),
      1e-6, 1e-4));
}

TEST(AugogradTest, LogSoftmaxBackward) {

  Tensor a = Tensor({2, 3, 4}, {0.2412, 0.4876, 0.5618, 0.6176, 0.6800, 0.9256,
                                0.5233, 0.6787, 0.3179, 0.1484, 0.4914, 0.8708,
                                0.7588, 0.5124, 0.4382, 0.3824, 0.3200, 0.0744,
                                0.4767, 0.3213, 0.6821, 0.8516, 0.5086, 0.1292},
                    true);

  Tensor target = ones({2, 3, 4});

  Tensor b = log_softmax(a, 0);

  // b.print();
  EXPECT_TRUE(b.strict_allclose(
      Tensor({2, 3, 4}, {-0.9851, -0.7056, -0.6333, -0.5824, -0.5293, -0.3555,
                         -0.6701, -0.5303, -0.8917, -1.1053, -0.7018, -0.3896,
                         -0.4675, -0.6808, -0.7569, -0.8176, -0.8893, -1.2067,
                         -0.7167, -0.8877, -0.5275, -0.4021, -0.6846, -1.1312}),
      1e-6, 1e-4));

  // result.print();
  Tensor c = sum(target * toytorch::log(b));

  // std::cout << debug::print_backward_graph(c);
  c.backward();

  EXPECT_TRUE(a.grad()->strict_allclose(
      Tensor({2, 3, 4}, {0.1627,  0.0079,  -0.0394, -0.0749, -0.1141, -0.2608,
                         -0.0149, -0.1133, 0.1154,  0.2182,  0.0055,  -0.2294,
                         -0.1627, -0.0079, 0.0394,  0.0749,  0.1141,  0.2608,
                         0.0149,  0.1133,  -0.1154, -0.2182, -0.0055, 0.2294}),
      1e-6, 1e-4));
}

TEST(AugogradTest, LogSoftmaxBackwardDim1) {

  Tensor a = Tensor({2, 3, 4}, {0.2412, 0.4876, 0.5618, 0.6176, 0.6800, 0.9256,
                                0.5233, 0.6787, 0.3179, 0.1484, 0.4914, 0.8708,
                                0.7588, 0.5124, 0.4382, 0.3824, 0.3200, 0.0744,
                                0.4767, 0.3213, 0.6821, 0.8516, 0.5086, 0.1292},
                    true);

  Tensor target = ones({2, 3, 4});

  Tensor b = log_softmax(a, 1);

  // b.print();
  EXPECT_TRUE(b.strict_allclose(
      Tensor({2, 3, 4}, {-1.2894, -1.1823, -1.0627, -1.2093, -0.8506, -0.7443,
                         -1.1012, -1.1482, -1.2127, -1.5215, -1.1331, -0.9561,
                         -0.9443, -1.1149, -1.1353, -0.9995, -1.3831, -1.5529,
                         -1.0968, -1.0606, -1.0210, -0.7757, -1.0649, -1.2527}),
      1e-6, 1e-4));

  // result.print();
  Tensor c = sum(target * toytorch::log(b));

  // std::cout << debug::print_backward_graph(c);
  c.backward();

  EXPECT_TRUE(a.grad()->strict_allclose(
      Tensor({2, 3, 4},
             {-1.0988e-02, 2.6856e-02,  2.8147e-03,  -8.1694e-03, 1.0078e-02,
              8.7580e-03,  6.6861e-05,  -5.8719e-04, 9.0978e-04,  -3.5614e-02,
              -2.8814e-03, 8.7569e-03,  1.5091e-02,  3.1210e-02,  -3.1072e-03,
              8.5556e-03,  -3.0436e-02, -4.4993e-02, 4.2561e-04,  6.3835e-03,
              1.5344e-02,  1.3783e-02,  2.6814e-03,  -1.4939e-02}),
      1e-6, 1e-4));
}