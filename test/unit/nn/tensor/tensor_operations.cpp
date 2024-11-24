#include "nn/tensor/tensor_operations.h"
#include <gtest/gtest.h>
#include "exception/exceptions.h"
#include "nn/tensor/tensor.h"
#include "nn/tensor/tensor_creator.h"

using namespace toytorch;

TEST(TensorOperationsTest, ArithmeticOperandUnchanged) {
  Tensor t0({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t2({3, 2}, {1, 2, 3, 4, 5, 6});

  Tensor t = t1 + t2;

  EXPECT_TRUE(t1 == t0);
  EXPECT_TRUE(t2 == t0);
}

TEST(TensorOperationsTest, ArithmeticOperandUnchangedAfterBroadcast) {
  Tensor t1({1, 2}, {1, 1});
  Tensor t2({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t_sum({3, 2}, {});

  Tensor t = t1 + t2;

  EXPECT_TRUE(t == Tensor({3, 2}, {2, 3, 4, 5, 6, 7}));
  EXPECT_TRUE(t1 == Tensor({1, 2}, {1, 1}));
  EXPECT_TRUE(t2 == Tensor({3, 2}, {1, 2, 3, 4, 5, 6}));
}

TEST(TensorOperationsTest, Matmul_mDmD_TransposedRandomValueResult) {
  // calculated with numpy
  Tensor t1({2, 3, 3, 2}, {14, 29, 20, 12, 9,  19, 26, 23, 16, 25, 28, 20,
                           14, 8,  19, 2,  25, 21, 13, 27, 28, 29, 22, 11,
                           24, 10, 29, 14, 2,  28, 13, 4,  18, 17, 3,  3});
  Tensor t2({3, 2, 4}, {4, 25, 4,  3,  3, 3,  26, 18, 8,  19, 29, 13,
                        6, 24, 18, 28, 2, 27, 5,  20, 21, 21, 21, 18});
  Tensor expect_result =
      Tensor({2, 3, 3, 4},
             {143, 437,  810,  564, 116, 536, 392,  276, 93,  282,  530,  369,
              346, 1046, 1168, 982, 278, 904, 914,  908, 344, 1012, 1172, 924,
              196, 546,  238,  424, 80,  555, 137,  416, 491, 1116, 566,  878,
              133, 406,  754,  525, 199, 787, 866,  606, 121, 583,  374,  264,
              252, 696,  876,  592, 316, 887, 1093, 769, 184, 710,  562,  810,
              110, 435,  149,  332, 393, 843, 447,  666, 69,  144,  78,   114});

  EXPECT_TRUE(matmul(t2.transpose(1, 2), t1.transpose(2, 3)) ==
              expect_result.transpose(2, 3));
}

TEST(TensorOperationsTest, ElementwiseUnaryOpExp) {
  Tensor t({2, 3}, {1, 1, 1, 1, 1, 1});

  Tensor result = toytorch::exp(t);
  result.print();
  EXPECT_TRUE(result == Tensor({2, 3}, {2.71828182, 2.71828183, 2.71828183,
                                        2.71828183, 2.71828183, 2.71828183}));
}

TEST(TensorOperationsTest, ElementwiseUnaryOpNeg) {
  Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});

  EXPECT_TRUE(toytorch::neg(t) == Tensor({2, 3}, {-1, -2, -3, -4, -5, -6}));
}

TEST(TensorOperationsTest, ElementwiseAdd) {
  Tensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t2({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t_sum({3, 2}, {2, 4, 6, 8, 10, 12});

  EXPECT_TRUE((t1 + t2) == t_sum);
}

TEST(TensorOperationsTest, ElementwiseMinus) {
  Tensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t2({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t_minus({3, 2}, {0, 0, 0, 0, 0, 0});

  EXPECT_TRUE((t1 - t2) == t_minus);
}

TEST(TensorOperationsTest, ElementwiseMul) {
  Tensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t2({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t_mul({3, 2}, {1, 4, 9, 16, 25, 36});

  EXPECT_TRUE((t1 * t2) == t_mul);
}

TEST(TensorOperationsTest, ElementwiseDiv) {
  Tensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t2({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t_div({3, 2}, {1, 1, 1, 1, 1, 1});

  EXPECT_TRUE((t1 / t2) == t_div);
}

TEST(TensorOperationsTest, ElementwisePow) {
  Tensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t2({3, 2}, {2, 2, 3, 4, 2, 1});

  EXPECT_TRUE((t1 ^ t2) == Tensor({3, 2}, {1, 4, 27, 256, 25, 6}));

  Tensor t3(2);
  EXPECT_TRUE((t3 ^ t2) == Tensor({3, 2}, {4, 4, 8, 16, 4, 2}));
  EXPECT_TRUE((t2 ^ t3) == Tensor({3, 2}, {4, 4, 9, 16, 4, 1}));

  Tensor t4(-1);
  EXPECT_TRUE((t2 ^ t4) == Tensor(1) / t2);
}

TEST(TensorOperationsTest, ElementwiseWithScalarAsOp1) {
  Tensor t1(1);
  Tensor t2({3, 2}, {1, 2, 3, 4, 5, 6});

  EXPECT_TRUE((t1 + t2) == Tensor({3, 2}, {2, 3, 4, 5, 6, 7}));
  EXPECT_TRUE((t1 - t2) == Tensor({3, 2}, {0, -1, -2, -3, -4, -5}));
  EXPECT_TRUE((t1 * t2) == Tensor({3, 2}, {1, 2, 3, 4, 5, 6}));
  EXPECT_TRUE((t1 / t2) ==
              Tensor({3, 2}, {1, 0.5, 1.0f / 3, 0.25, 0.2, 1.0f / 6}));

  Tensor t3(2);
  EXPECT_TRUE((t2 ^ t3) == Tensor({3, 2}, {1, 4, 9, 16, 25, 36}));
}

TEST(TensorTest, ElementwiseWithScalarAsOp2) {
  Tensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t2(1);

  Tensor t_sum({3, 2}, {2, 3, 4, 5, 6, 7});
  Tensor t_sub({3, 2}, {0, 1, 2, 3, 4, 5});
  Tensor t_mul({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t_div({3, 2}, {1, 2, 3, 4, 5, 6});

  EXPECT_TRUE((t1 + t2) == t_sum);
  EXPECT_TRUE((t1 - t2) == t_sub);
  EXPECT_TRUE((t1 * t2) == t_mul);
  EXPECT_TRUE((t1 / t2) == t_div);
}

TEST(TensorOperationsTest, Matmul_1D1D_Compatible) {
  Tensor t1({3}, {2, 2, 2});
  Tensor t2({3}, {3, 3, 3});

  Tensor result(18);

  EXPECT_TRUE(matmul(t1, t2) == result);
}

TEST(TensorOperationsTest, Matmul_1D1D_Incompatible) {
  Tensor t1({3}, {2, 2, 2});
  Tensor t2({2}, {3, 2});

  EXPECT_THROW(matmul(t1, t2), ExceptionInvalidArgument);
}

TEST(TensorOperationsTest, Matmul_1D2D_Compatible) {
  Tensor t1({3}, {2, 2, 2});
  Tensor t2({3, 1}, {3, 3, 3});

  Tensor t0({1}, 18);

  Tensor result = matmul(t1, t2);
  EXPECT_TRUE(result.shape() == TensorShape({1}));
  EXPECT_TRUE(result[0] == 18);
  EXPECT_TRUE(result == t0);
}

TEST(TensorOperationsTest, Matmul_1D2D_Incompatible) {
  Tensor t1({3}, {2, 2, 2});
  Tensor t2({2, 1}, {3, 2});

  EXPECT_THROW(matmul(t1, t2), ExceptionInvalidArgument);
}

TEST(TensorOperationsTest, Matmul_2D1D_Compatible) {
  Tensor t1({3}, {2, 2, 2});
  Tensor t2({1, 3}, {3, 3, 3});

  Tensor t0({1}, 18);

  Tensor result = matmul(t2, t1);
  EXPECT_TRUE(result.shape() == TensorShape({1}));
  EXPECT_TRUE(result[0] == 18);
  EXPECT_TRUE(result == t0);
}

TEST(TensorOperationsTest, Matmul_2D1D_Incompatible) {
  Tensor t1({3}, {2, 2, 2});
  Tensor t2({2, 1}, {3, 2});

  EXPECT_THROW(matmul(t2, t1), ExceptionInvalidArgument);
}

TEST(TensorOperationsTest, Matmul_2D2D_Compatible) {
  Tensor t1({3, 2}, {4, 13, 29, 29, 20, 28});
  Tensor t2({2, 4}, {26, 14, 25, 18, 28, 28, 18, 17});

  Tensor expected_result = Tensor({3, 4}, {468, 420, 334, 293, 1566, 1218, 1247,
                                           1015, 1304, 1064, 1004, 836});

  EXPECT_TRUE(matmul(t1, t2) == expected_result);
}

TEST(TensorOperationsTest, Matmul_2D2D_Incompatible) {
  Tensor t1({3, 2}, {2, 2, 2, 2, 2, 2});
  Tensor t2({3, 2}, {3, 3, 3, 3, 3, 3});

  EXPECT_THROW(matmul(t1, t2), ExceptionInvalidArgument);
}

TEST(TensorOperationsTest, Matmul_3D2D_Compatible) {

  Tensor t1({2, 3, 2}, {12, 25, 16, 9, 16, 27, 18, 14, 24, 10, 23, 28});
  Tensor t2({1, 2, 4}, {28, 24, 6, 24, 17, 10, 7, 28});

  Tensor expected_result =
      Tensor({2, 3, 4},
             {761, 538, 247, 988, 601, 474, 159, 636, 907,  654, 285, 1140,
              742, 572, 206, 824, 842, 676, 214, 856, 1120, 832, 334, 1336});

  EXPECT_TRUE(matmul(t1, t2) == expected_result);
}

TEST(TensorOperationsTest, Matmul_3D2D_ShapeIncompatible) {

  Tensor t1({2, 3, 2});
  Tensor t2({1, 3, 4});

  EXPECT_THROW(matmul(t1, t2), ExceptionTensorShapeIncompatible);
}

TEST(TensorOperationsTest, Matmul_3D2D_BatchIncompatible) {

  Tensor t1({2, 3, 2});
  Tensor t2({3, 2, 4});

  EXPECT_THROW(matmul(t1, t2), ExceptionTensorShapeIncompatible);
}

TEST(TensorOperationsTest, Matmul_mDmD_BatchCompatible) {

  Tensor t1({2, 3, 3, 2}, std::vector<float>(36, 8));
  Tensor t2({3, 2, 4}, std::vector<float>(24, 5));

  Tensor t3 = matmul(t1, t2);
  EXPECT_TRUE(t3 == Tensor({2, 3, 3, 4}, std::vector<float>(72, 80)));
}

TEST(TensorOperationsTest, Matmul_mDmD_RandomValueResult) {
  // calculated with numpy
  Tensor t1({2, 3, 3, 2}, {14, 29, 20, 12, 9,  19, 26, 23, 16, 25, 28, 20,
                           14, 8,  19, 2,  25, 21, 13, 27, 28, 29, 22, 11,
                           24, 10, 29, 14, 2,  28, 13, 4,  18, 17, 3,  3});
  Tensor t2({3, 2, 4}, {4, 25, 4,  3,  3, 3,  26, 18, 8,  19, 29, 13,
                        6, 24, 18, 28, 2, 27, 5,  20, 21, 21, 21, 18});
  Tensor expect_result =
      Tensor({2, 3, 3, 4},
             {143, 437,  810,  564, 116, 536, 392,  276, 93,  282,  530,  369,
              346, 1046, 1168, 982, 278, 904, 914,  908, 344, 1012, 1172, 924,
              196, 546,  238,  424, 80,  555, 137,  416, 491, 1116, 566,  878,
              133, 406,  754,  525, 199, 787, 866,  606, 121, 583,  374,  264,
              252, 696,  876,  592, 316, 887, 1093, 769, 184, 710,  562,  810,
              110, 435,  149,  332, 393, 843, 447,  666, 69,  144,  78,   114});

  EXPECT_TRUE(matmul(t1, t2) == expect_result);
}

TEST(TensorOperationsTest, Matmul_2D2D_DiffLayoutTransposedMatmul) {

  Tensor t1({2, 3}, {4, 13, 29, 29, 20, 28});
  Tensor t1_t({3, 2}, {4, 29, 13, 20, 29, 28});

  Tensor t2({2, 4}, {26, 14, 25, 18, 28, 28, 18, 17});

  EXPECT_TRUE(matmul(t1.transpose(), t2) == matmul(t1_t, t2));
}

TEST(TensorOperationsTest, Matmul_2D2D_TransposedRandomValueResult) {

  Tensor t1({3, 2}, {4, 13, 29, 29, 20, 28});
  Tensor t2({2, 4}, {26, 14, 25, 18, 28, 28, 18, 17});

  Tensor expected_result = Tensor({3, 4}, {468, 420, 334, 293, 1566, 1218, 1247,
                                           1015, 1304, 1064, 1004, 836});

  EXPECT_TRUE(matmul(t2.transpose(), t1.transpose()) ==
              expected_result.transpose());
}

TEST(TensorOperationsTest, GtNoBroadcast) {
  Tensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t2({3, 2}, {0, 3, 2, 4, 3, 7});

  EXPECT_TRUE((t1 > t2) == Tensor({3, 2}, {1, 0, 1, 0, 1, 0}));
  EXPECT_TRUE((t2 > t1) == Tensor({3, 2}, {0, 1, 0, 0, 0, 1}));
}

TEST(TensorOperationsTest, GtWithBroadcast) {
  Tensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t2({2}, {0, 3});

  EXPECT_TRUE((t1 > t2) == Tensor({3, 2}, {1, 0, 1, 1, 1, 1}));
  EXPECT_TRUE((t2 > t1) == Tensor({3, 2}, {0, 1, 0, 0, 0, 0}));
}

TEST(TensorOperationsTest, LtNoBroadcast) {
  Tensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t2({3, 2}, {0, 3, 2, 4, 3, 7});

  EXPECT_TRUE((t2 < t1) == Tensor({3, 2}, {1, 0, 1, 0, 1, 0}));
  EXPECT_TRUE((t1 < t2) == Tensor({3, 2}, {0, 1, 0, 0, 0, 1}));
}

TEST(TensorOperationsTest, LtWithBroadcast) {
  Tensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t2({2}, {0, 3});

  EXPECT_TRUE((t2 < t1) == Tensor({3, 2}, {1, 0, 1, 1, 1, 1}));
  EXPECT_TRUE((t1 < t2) == Tensor({3, 2}, {0, 1, 0, 0, 0, 0}));
}

TEST(TensorOperationsTest, GeNoBroadcast) {
  Tensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t2({3, 2}, {0, 3, 2, 4, 3, 7});

  EXPECT_TRUE((t1 >= t2) == Tensor({3, 2}, {1, 0, 1, 1, 1, 0}));
  EXPECT_TRUE((t2 >= t1) == Tensor({3, 2}, {0, 1, 0, 1, 0, 1}));
}

TEST(TensorOperationsTest, LeNoBroadcast) {
  Tensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t2({3, 2}, {0, 3, 2, 4, 3, 7});

  EXPECT_TRUE((t2 <= t1) == Tensor({3, 2}, {1, 0, 1, 1, 1, 0}));
  EXPECT_TRUE((t1 <= t2) == Tensor({3, 2}, {0, 1, 0, 1, 0, 1}));
}

TEST(TensorOperationsTest, WhereNoBroadcast) {
  Tensor condition({2, 3}, {1, 0, 0, 1, 1, 1});
  Tensor t1({2, 3}, {2, 2, 2, 2, 2, 2});
  Tensor t2({2, 3}, {4, 4, 4, 4, 4, 4});

  Tensor result = where(condition, t1, t2);
  EXPECT_TRUE(result == Tensor({2, 3}, {2, 4, 4, 2, 2, 2}));
}

TEST(TensorOperationsTest, WhereWithBroadcast) {
  Tensor condition({3}, {1, 0, 0});
  Tensor t1({2, 2, 3}, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
  Tensor t2({2, 3}, {4, 4, 4, 4, 4, 4});

  Tensor result = where(condition, t1, t2);
  EXPECT_TRUE(result ==
              Tensor({2, 2, 3}, {2, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 4}));
}

TEST(TensorOperationsTest, WhereConditionGt) {
  Tensor t1({2, 3}, {-1, 2, -2, -3, 2, 3});

  Tensor result = where(0 > t1, neg(t1), t1);
  EXPECT_TRUE(result == Tensor({2, 3}, {1, 2, 2, 3, 2, 3}));
}

TEST(TensorOperationsTest, SumAll) {
  Tensor t1 = ones({4, 3});

  Tensor result = t1.sum();
  EXPECT_TRUE(result == Tensor(12));
}

TEST(TensorOperationsTest, SumOneAxis) {
  Tensor t1 = ones({2, 3});

  Tensor result = t1.sum(0, true);
  EXPECT_TRUE(result == Tensor({1, 3}, {2, 2, 2}));
}

TEST(TensorOperationsTest, SumWithMultiAxis) {
  Tensor t1 = ones({2, 2, 3});

  Tensor result = t1.sum({0, 1}, true);
  EXPECT_TRUE(result == Tensor({1, 1, 3}, {4, 4, 4}));
}

TEST(TensorOperationsTest, Cat) {
  Tensor t1 = ones({2, 2, 3});
  Tensor t2 = ones({1, 2, 3});

  EXPECT_THROW(cat({t1, t2}, 3), ExceptionInvalidArgument);
  EXPECT_THROW(cat({t1, t2}, -4), ExceptionInvalidArgument);

  EXPECT_THROW(cat({t1, t2}, 1), ExceptionTensorShapeIncompatible);

  EXPECT_TRUE(cat({t1, t2}, 0) == ones({3, 2, 3}));
  EXPECT_TRUE(cat({t1, t2}, -3) == ones({3, 2, 3}));

  Tensor t3 = Tensor({2, 2, 1}, {
                                    1,
                                    2,
                                    3,
                                    4,
                                });
  Tensor t4 = Tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  Tensor t5 = Tensor({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  EXPECT_TRUE(cat({t3, t4, t5}, -1) ==
              Tensor({2, 2, 6}, {1, 1, 2, 1, 2, 3, 2, 3, 4, 4,  5,  6,
                                 3, 5, 6, 7, 8, 9, 4, 7, 8, 10, 11, 12}));
}

TEST(TensorOperationsTest, CatBackwardThrow) {
  Tensor t1 = ones({2, 2, 3}, true);
  Tensor t2 = ones({1, 2, 3});

  // After implemented, we will remove this test
  EXPECT_THROW(cat({t1, t2}, 0), ExceptionOpBackwardNotImplemented);
}

TEST(TensorOperationsTest, squeeze) {
  Tensor t1 = ones({2, 3, 3});
  Tensor t2 = ones({2, 1, 3});

  EXPECT_THROW(squeeze(t1, 0), ExceptionInvalidArgument);
  EXPECT_THROW(squeeze(t1, 1), ExceptionInvalidArgument);
  EXPECT_THROW(squeeze(t1, 2), ExceptionInvalidArgument);

  EXPECT_THROW(squeeze(t2, 0), ExceptionInvalidArgument);
  EXPECT_TRUE(squeeze(t2, 1) == ones({2, 3}));
  EXPECT_TRUE(squeeze(t2, -2) == ones({2, 3}));

  Tensor t3 = ones({1, 2, 3}, true);

  EXPECT_THROW(squeeze(t3, 0), ExceptionOpBackwardNotImplemented);
}

TEST(TensorOperationsTest, unsqueeze) {
  Tensor t1 = ones({2, 3});
  EXPECT_THROW(unsqueeze(t1, 3), ExceptionInvalidArgument);
  EXPECT_THROW(unsqueeze(t1, -4), ExceptionInvalidArgument);

  EXPECT_TRUE(unsqueeze(t1, -3) == ones({1, 2, 3}));
  EXPECT_TRUE(unsqueeze(t1, 0) == ones({1, 2, 3}));
  EXPECT_TRUE(unsqueeze(t1, -1) == ones({2, 3, 1}));
  EXPECT_TRUE(unsqueeze(t1, 2) == ones({2, 3, 1}));

  Tensor t2 = ones({2, 3}, true);
  EXPECT_THROW(unsqueeze(t2, 0), ExceptionOpBackwardNotImplemented);
}

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
