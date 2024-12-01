#include "nn/operations/tensor_operations.h"
#include <gtest/gtest.h>
#include "nn/exceptions/exceptions.h"
#include "nn/operations/common_types.h"
#include "nn/operations/losses.h"
#include "nn/tensor/tensor.h"
#include "nn/tensor/tensor_creator.h"
#include "nn/utils/print_utils.h"

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

TEST(TensorOperationsTest, transpose) {
  Tensor t1 = arange(0, 24).view({2, 3, 4});

  EXPECT_TRUE(
      t1.transpose(0, 1) ==
      Tensor({3, 2, 4}, {0,  1,  2,  3,  12, 13, 14, 15, 4,  5,  6,  7,
                         16, 17, 18, 19, 8,  9,  10, 11, 20, 21, 22, 23}));
  EXPECT_TRUE(
      t1.transpose(2, 1) ==
      Tensor({2, 4, 3}, {0,  4,  8,  1,  5,  9,  2,  6,  10, 3,  7,  11,
                         12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23}));
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

TEST(TensorOperationsTest, unfoldNoOverlap) {
  Tensor t1 = arange(0, 2 * 3 * 4 * 6).view({2, 3, 4, 6});
  ASSERT_TRUE(t1.shape() == TensorShape({2, 3, 4, 6}));
  ASSERT_TRUE(t1.strides() == TensorShape({72, 24, 6, 1}));

  int kernel_h = 2;
  int kernel_w = 2;
  int stride = 2;

  Tensor t2 = t1.unfold(2, kernel_h, stride);
  // std::cout << t2.shape();
  // std::cout << t2.strides();

  EXPECT_TRUE(t2.shape() == TensorShape({2, 3, 2, 6, 2}));
  EXPECT_TRUE(t2.strides() == TensorShape({72, 24, 12, 1, 6}));

  Tensor t3 = t2.unfold(3, kernel_w, stride);
  EXPECT_TRUE(t3.shape() == TensorShape({2, 3, 2, 3, 2, 2}));
  EXPECT_TRUE(t3.strides() == TensorShape({72, 24, 12, 2, 6, 1}));

  EXPECT_TRUE(
      t3 ==
      Tensor(
          {2, 3, 2, 3, 2, 2},
          {0,   1,   6,   7,   2,   3,   8,   9,   4,   5,   10,  11,  12,  13,
           18,  19,  14,  15,  20,  21,  16,  17,  22,  23,  24,  25,  30,  31,
           26,  27,  32,  33,  28,  29,  34,  35,  36,  37,  42,  43,  38,  39,
           44,  45,  40,  41,  46,  47,  48,  49,  54,  55,  50,  51,  56,  57,
           52,  53,  58,  59,  60,  61,  66,  67,  62,  63,  68,  69,  64,  65,
           70,  71,  72,  73,  78,  79,  74,  75,  80,  81,  76,  77,  82,  83,
           84,  85,  90,  91,  86,  87,  92,  93,  88,  89,  94,  95,  96,  97,
           102, 103, 98,  99,  104, 105, 100, 101, 106, 107, 108, 109, 114, 115,
           110, 111, 116, 117, 112, 113, 118, 119, 120, 121, 126, 127, 122, 123,
           128, 129, 124, 125, 130, 131, 132, 133, 138, 139, 134, 135, 140, 141,
           136, 137, 142, 143}));
}

TEST(TensorOperationsTest, unfoldOverlap) {
  Tensor t1 = arange(0, 2 * 3 * 4 * 6).view({2, 3, 4, 6});
  ASSERT_TRUE(t1.shape() == TensorShape({2, 3, 4, 6}));
  ASSERT_TRUE(t1.strides() == TensorShape({72, 24, 6, 1}));

  int kernel_h = 2;
  int kernel_w = 2;
  int stride = 1;

  Tensor t2 = t1.unfold(2, kernel_h, stride);
  // std::cout << t2.shape();
  // std::cout << t2.strides();

  EXPECT_TRUE(t2.shape() == TensorShape({2, 3, 3, 6, 2}));
  EXPECT_TRUE(t2.strides() == TensorShape({72, 24, 6, 1, 6}));

  Tensor t3 = t2.unfold(3, kernel_w, stride);
  EXPECT_TRUE(t3.shape() == TensorShape({2, 3, 3, 5, 2, 2}));
  EXPECT_TRUE(t3.strides() == TensorShape({72, 24, 6, 1, 6, 1}));

  EXPECT_TRUE(
      t3 ==
      Tensor(
          {2, 3, 3, 5, 2, 2},
          {0,   1,   6,   7,   1,   2,   7,   8,   2,   3,   8,   9,   3,   4,
           9,   10,  4,   5,   10,  11,  6,   7,   12,  13,  7,   8,   13,  14,
           8,   9,   14,  15,  9,   10,  15,  16,  10,  11,  16,  17,  12,  13,
           18,  19,  13,  14,  19,  20,  14,  15,  20,  21,  15,  16,  21,  22,
           16,  17,  22,  23,  24,  25,  30,  31,  25,  26,  31,  32,  26,  27,
           32,  33,  27,  28,  33,  34,  28,  29,  34,  35,  30,  31,  36,  37,
           31,  32,  37,  38,  32,  33,  38,  39,  33,  34,  39,  40,  34,  35,
           40,  41,  36,  37,  42,  43,  37,  38,  43,  44,  38,  39,  44,  45,
           39,  40,  45,  46,  40,  41,  46,  47,  48,  49,  54,  55,  49,  50,
           55,  56,  50,  51,  56,  57,  51,  52,  57,  58,  52,  53,  58,  59,
           54,  55,  60,  61,  55,  56,  61,  62,  56,  57,  62,  63,  57,  58,
           63,  64,  58,  59,  64,  65,  60,  61,  66,  67,  61,  62,  67,  68,
           62,  63,  68,  69,  63,  64,  69,  70,  64,  65,  70,  71,  72,  73,
           78,  79,  73,  74,  79,  80,  74,  75,  80,  81,  75,  76,  81,  82,
           76,  77,  82,  83,  78,  79,  84,  85,  79,  80,  85,  86,  80,  81,
           86,  87,  81,  82,  87,  88,  82,  83,  88,  89,  84,  85,  90,  91,
           85,  86,  91,  92,  86,  87,  92,  93,  87,  88,  93,  94,  88,  89,
           94,  95,  96,  97,  102, 103, 97,  98,  103, 104, 98,  99,  104, 105,
           99,  100, 105, 106, 100, 101, 106, 107, 102, 103, 108, 109, 103, 104,
           109, 110, 104, 105, 110, 111, 105, 106, 111, 112, 106, 107, 112, 113,
           108, 109, 114, 115, 109, 110, 115, 116, 110, 111, 116, 117, 111, 112,
           117, 118, 112, 113, 118, 119, 120, 121, 126, 127, 121, 122, 127, 128,
           122, 123, 128, 129, 123, 124, 129, 130, 124, 125, 130, 131, 126, 127,
           132, 133, 127, 128, 133, 134, 128, 129, 134, 135, 129, 130, 135, 136,
           130, 131, 136, 137, 132, 133, 138, 139, 133, 134, 139, 140, 134, 135,
           140, 141, 135, 136, 141, 142, 136, 137, 142, 143}));
}

TEST(TensorOperationsTest, unfoldNotAlign) {
  Tensor t1 = arange(0, 2 * 3 * 5 * 7).view({2, 3, 5, 7});

  ASSERT_TRUE(t1.shape() == TensorShape({2, 3, 5, 7}));
  ASSERT_TRUE(t1.strides() == TensorShape({105, 35, 7, 1}));

  int kernel_h = 2;
  int kernel_w = 3;
  int stride = 2;

  Tensor t2 = t1.unfold(2, kernel_h, stride);
  // std::cout << t2.shape();
  // std::cout << t2.strides();

  EXPECT_TRUE(t2.shape() == TensorShape({2, 3, 2, 7, 2}));
  EXPECT_TRUE(t2.strides() == TensorShape({105, 35, 14, 1, 7}));

  Tensor t3 = t2.unfold(3, kernel_w, stride);
  EXPECT_TRUE(t3.shape() == TensorShape({2, 3, 2, 3, 2, 3}));
  EXPECT_TRUE(t3.strides() == TensorShape({105, 35, 14, 2, 7, 1}));

  EXPECT_TRUE(
      t3 ==
      Tensor(
          {2, 3, 2, 3, 2, 3},
          {0,   1,   2,   7,   8,   9,   2,   3,   4,   9,   10,  11,  4,   5,
           6,   11,  12,  13,  14,  15,  16,  21,  22,  23,  16,  17,  18,  23,
           24,  25,  18,  19,  20,  25,  26,  27,  35,  36,  37,  42,  43,  44,
           37,  38,  39,  44,  45,  46,  39,  40,  41,  46,  47,  48,  49,  50,
           51,  56,  57,  58,  51,  52,  53,  58,  59,  60,  53,  54,  55,  60,
           61,  62,  70,  71,  72,  77,  78,  79,  72,  73,  74,  79,  80,  81,
           74,  75,  76,  81,  82,  83,  84,  85,  86,  91,  92,  93,  86,  87,
           88,  93,  94,  95,  88,  89,  90,  95,  96,  97,  105, 106, 107, 112,
           113, 114, 107, 108, 109, 114, 115, 116, 109, 110, 111, 116, 117, 118,
           119, 120, 121, 126, 127, 128, 121, 122, 123, 128, 129, 130, 123, 124,
           125, 130, 131, 132, 140, 141, 142, 147, 148, 149, 142, 143, 144, 149,
           150, 151, 144, 145, 146, 151, 152, 153, 154, 155, 156, 161, 162, 163,
           156, 157, 158, 163, 164, 165, 158, 159, 160, 165, 166, 167, 175, 176,
           177, 182, 183, 184, 177, 178, 179, 184, 185, 186, 179, 180, 181, 186,
           187, 188, 189, 190, 191, 196, 197, 198, 191, 192, 193, 198, 199, 200,
           193, 194, 195, 200, 201, 202}));
}

TEST(TensorOperationsTest, unfold) {
  Tensor t1 = arange(0, 2 * 3 * 4 * 6).view({2, 3, 4, 6});
  ASSERT_TRUE(t1.shape() == TensorShape({2, 3, 4, 6}));
  ASSERT_TRUE(t1.strides() == TensorShape({72, 24, 6, 1}));

  int kernel_h = 2;
  int kernel_w = 2;
  int stride = 2;

  Tensor t2 = t1.unfold(2, kernel_h, stride);
  // std::cout << t2.shape();
  // std::cout << t2.strides();

  EXPECT_TRUE(t2.shape() == TensorShape({2, 3, 2, 6, 2}));
  EXPECT_TRUE(t2.strides() == TensorShape({72, 24, 12, 1, 6}));

  Tensor t3 = t2.unfold(3, kernel_w, stride);
  EXPECT_TRUE(t3.shape() == TensorShape({2, 3, 2, 3, 2, 2}));
  EXPECT_TRUE(t3.strides() == TensorShape({72, 24, 12, 2, 6, 1}));

  EXPECT_TRUE(
      t3 ==
      Tensor(
          {2, 3, 2, 3, 2, 2},
          {0,   1,   6,   7,   2,   3,   8,   9,   4,   5,   10,  11,  12,  13,
           18,  19,  14,  15,  20,  21,  16,  17,  22,  23,  24,  25,  30,  31,
           26,  27,  32,  33,  28,  29,  34,  35,  36,  37,  42,  43,  38,  39,
           44,  45,  40,  41,  46,  47,  48,  49,  54,  55,  50,  51,  56,  57,
           52,  53,  58,  59,  60,  61,  66,  67,  62,  63,  68,  69,  64,  65,
           70,  71,  72,  73,  78,  79,  74,  75,  80,  81,  76,  77,  82,  83,
           84,  85,  90,  91,  86,  87,  92,  93,  88,  89,  94,  95,  96,  97,
           102, 103, 98,  99,  104, 105, 100, 101, 106, 107, 108, 109, 114, 115,
           110, 111, 116, 117, 112, 113, 118, 119, 120, 121, 126, 127, 122, 123,
           128, 129, 124, 125, 130, 131, 132, 133, 138, 139, 134, 135, 140, 141,
           136, 137, 142, 143}));
}

TEST(TensorOperationsTest, slice) {
  Tensor t1 = arange(0, 60).view({3, 4, 5});

  Tensor t2 = t1.slice(1, 1, 3);
  EXPECT_TRUE(t2 ==
              Tensor({3, 2, 5}, {5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                 45, 46, 47, 48, 49, 50, 51, 52, 53, 54}));

  Tensor t3 = t2.slice(2, 2, 4);
  EXPECT_TRUE(
      t3 == Tensor({3, 2, 2}, {7, 8, 12, 13, 27, 28, 32, 33, 47, 48, 52, 53}));

  Tensor t4 = t1.slice(0, 0, 3).slice(1, 0, 4).slice(2, 0, 5);

  EXPECT_TRUE(t4 == t1);

  // dim out of range
  EXPECT_THROW(t1.slice(3, 1, 2), ExceptionInvalidArgument);

  // start > end
  EXPECT_THROW(t1.slice(0, 4, 2), ExceptionInvalidArgument);

  // start out of range
  EXPECT_THROW(t1.slice(0, 4, 6), ExceptionInvalidArgument);

  // end out of range
  EXPECT_THROW(t1.slice(0, 1, 4), ExceptionInvalidArgument);
}

TEST(TensorOperationsTest, flip) {

  Tensor t = arange(0, 120).view({2, 3, 4, 5});

  Tensor result = flip(t, {1, 2});
  // result.print();

  EXPECT_TRUE(
      result ==
      Tensor(
          {2, 3, 4, 5},
          {55,  56,  57,  58,  59,  50,  51,  52,  53,  54,  45,  46,  47,  48,
           49,  40,  41,  42,  43,  44,  35,  36,  37,  38,  39,  30,  31,  32,
           33,  34,  25,  26,  27,  28,  29,  20,  21,  22,  23,  24,  15,  16,
           17,  18,  19,  10,  11,  12,  13,  14,  5,   6,   7,   8,   9,   0,
           1,   2,   3,   4,   115, 116, 117, 118, 119, 110, 111, 112, 113, 114,
           105, 106, 107, 108, 109, 100, 101, 102, 103, 104, 95,  96,  97,  98,
           99,  90,  91,  92,  93,  94,  85,  86,  87,  88,  89,  80,  81,  82,
           83,  84,  75,  76,  77,  78,  79,  70,  71,  72,  73,  74,  65,  66,
           67,  68,  69,  60,  61,  62,  63,  64}));

  result = flip(t, {2, 3});
  // result.print();

  EXPECT_TRUE(
      result ==
      Tensor(
          {2, 3, 4, 5},
          {19,  18,  17,  16,  15,  14,  13,  12,  11,  10,  9,   8,   7,   6,
           5,   4,   3,   2,   1,   0,   39,  38,  37,  36,  35,  34,  33,  32,
           31,  30,  29,  28,  27,  26,  25,  24,  23,  22,  21,  20,  59,  58,
           57,  56,  55,  54,  53,  52,  51,  50,  49,  48,  47,  46,  45,  44,
           43,  42,  41,  40,  79,  78,  77,  76,  75,  74,  73,  72,  71,  70,
           69,  68,  67,  66,  65,  64,  63,  62,  61,  60,  99,  98,  97,  96,
           95,  94,  93,  92,  91,  90,  89,  88,  87,  86,  85,  84,  83,  82,
           81,  80,  119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108,
           107, 106, 105, 104, 103, 102, 101, 100}));

  result = flip(t, {0, 1, 2, 3});
  // result.print();

  EXPECT_TRUE(
      result ==
      Tensor(
          {2, 3, 4, 5},
          {119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106,
           105, 104, 103, 102, 101, 100, 99,  98,  97,  96,  95,  94,  93,  92,
           91,  90,  89,  88,  87,  86,  85,  84,  83,  82,  81,  80,  79,  78,
           77,  76,  75,  74,  73,  72,  71,  70,  69,  68,  67,  66,  65,  64,
           63,  62,  61,  60,  59,  58,  57,  56,  55,  54,  53,  52,  51,  50,
           49,  48,  47,  46,  45,  44,  43,  42,  41,  40,  39,  38,  37,  36,
           35,  34,  33,  32,  31,  30,  29,  28,  27,  26,  25,  24,  23,  22,
           21,  20,  19,  18,  17,  16,  15,  14,  13,  12,  11,  10,  9,   8,
           7,   6,   5,   4,   3,   2,   1,   0}));
}
