#include "nn/operations/matrix.h"
#include "nn/exceptions/exceptions.h"
#include <gtest/gtest.h>

using namespace toytorch;

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

TEST(TensorOperationsTest, Matmul_1D1D_Compatible) {
  Tensor t1({3}, {2, 2, 2});
  Tensor t2({3}, {3, 3, 3});

  Tensor result(18);

  EXPECT_TRUE(matmul(t1, t2) == result);

  Tensor t3({2, 4}, {2, 2, 2, 2, 2, 2, 2, 2});
  Tensor t4({2, 4}, {5, 5, 5, 5, 5, 5, 5, 5});

  EXPECT_TRUE(matmul(t3.view({-1}), t4.view({-1})) == Tensor(80));
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
