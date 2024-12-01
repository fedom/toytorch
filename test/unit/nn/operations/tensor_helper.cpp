#include <gtest/gtest.h>
#include "nn/operations/tensor_helper.h"
#include "nn/exceptions/exceptions.h"

using namespace toytorch;

TEST(TensorHelperTest, CheckBroadcastableWithOneScalar) {
  Tensor t1(1);
  Tensor t2({1, 2});

  EXPECT_TRUE(TensorHelper::are_tensors_broadcastable(t1, t2));
  EXPECT_TRUE(TensorHelper::are_tensors_broadcastable(t2, t1));
}

TEST(TensorHelperTest, CheckBroadcastableWithBothScalar) {
  Tensor t1(1);
  Tensor t2(2);

  EXPECT_TRUE(TensorHelper::are_tensors_broadcastable(t1, t2));
  EXPECT_TRUE(TensorHelper::are_tensors_broadcastable(t2, t1));
}

TEST(TensorHelperTest, CheckBroadcastableAlignedWithOnes) {

  Tensor t1({3, 1});
  Tensor t2({1, 2});
  EXPECT_TRUE(TensorHelper::are_tensors_broadcastable(t1, t2));
}

TEST(TensorHelperTest, CheckBroadcastableNotAligned) {
  Tensor t1({3, 2});
  Tensor t2({2, 3});
  EXPECT_FALSE(TensorHelper::are_tensors_broadcastable(t1, t2));
}

TEST(TensorHelperTest, CheckBroadcastableDiffDimAligned) {
  Tensor t1({3, 3});
  Tensor t2({2, 3, 3});
  EXPECT_TRUE(TensorHelper::are_tensors_broadcastable(t1, t2));
}

TEST(TensorHelperTest, CheckBroadcastableDiffDimNotAligned) {
  Tensor t1({2, 3});
  Tensor t2({2, 3, 3});
  EXPECT_FALSE(TensorHelper::are_tensors_broadcastable(t1, t2));
}

TEST(TensorHelperTest, CheckBroadcastableWithInvalidSkipDim) {
  Tensor t1({2, 3, 3});
  Tensor t2({2, 3, 3});

  EXPECT_THROW(TensorHelper::are_tensors_broadcastable(t1, t2, 1),
               ExceptionInvalidArgument);
}

TEST(TensorHelperTest, CheckBroadcastableWithValidSkipDim) {
  Tensor t1({2, 3, 4});
  Tensor t2({2, 4, 5});
  EXPECT_TRUE(TensorHelper::are_tensors_broadcastable(t1, t2, 2));

  Tensor t3({1, 3, 4});
  Tensor t4({2, 4, 5});
  EXPECT_TRUE(TensorHelper::are_tensors_broadcastable(t3, t4, 2));

  Tensor t5({3, 3, 4});
  Tensor t6({2, 4, 5});
  EXPECT_FALSE(TensorHelper::are_tensors_broadcastable(t5, t6, 2));

  Tensor t7({5, 2, 3, 4});
  Tensor t8({2, 4, 5});
  EXPECT_TRUE(TensorHelper::are_tensors_broadcastable(t7, t8, 2));
}

TEST(TensorHelperTest, BroadcastTensorsWithOneScalar) {
  Tensor t1(1);
  Tensor t2({1, 3, 1});

  TensorHelper::broadcast_tensors(t1, t2, 0);

  Tensor result({1, 3, 1});

  EXPECT_TRUE(t1.shape() == t2.shape());
  EXPECT_TRUE(t1.shape() == result.shape());
}

TEST(TensorHelperTest, BroadcastTensorsWithBothScalar) {
  Tensor t1(1);
  Tensor t2(2);

  TensorHelper::broadcast_tensors(t1, t2, 0);

  EXPECT_TRUE(t1.shape() == t2.shape());
  EXPECT_TRUE(t1.shape() == TensorShape({}));
}

TEST(TensorHelperTest, BroadcastTensorsWithOnes) {
  Tensor t1({3, 1, 3});
  Tensor t2({1, 3, 1});

  TensorHelper::broadcast_tensors(t1, t2, 0);

  Tensor result({3, 3, 3});

  EXPECT_TRUE(t1.shape() == t2.shape());
  EXPECT_TRUE(t1.shape() == result.shape());
}

TEST(TensorHelperTest, BroadcastTensorsDiffDimAligned) {
  Tensor t1({3, 3});
  Tensor t2({2, 3, 3});

  TensorHelper::broadcast_tensors(t1, t2, 0);

  Tensor result({2, 3, 3});
  EXPECT_TRUE(t1.shape() == t2.shape());
  EXPECT_TRUE(t1.shape() == result.shape());
}

TEST(TensorHelperTest, BroadcastTensorsRawDataIdentical) {
  Tensor t1(TensorShape({3}));
  Tensor t2({2, 3, 1});

  Tensor t1_broadcasted(t1);
  Tensor t2_broadcasted(t2);

  TensorHelper::broadcast_tensors(t1_broadcasted, t2_broadcasted, 0);

  Tensor result({2, 3, 3});
  EXPECT_TRUE(t1_broadcasted.shape() == t2_broadcasted.shape());
  EXPECT_TRUE(t1_broadcasted.shape() == result.shape());

  EXPECT_TRUE(t1.at(TensorIndices({0})) ==
              t1_broadcasted.at(TensorIndices({0, 0, 0})));
  EXPECT_TRUE(t1.at(TensorIndices({0})) ==
              t1_broadcasted.at(TensorIndices({0, 1, 0})));
  EXPECT_TRUE(t1.at(TensorIndices({0})) ==
              t1_broadcasted.at(TensorIndices({1, 0, 0})));
  EXPECT_TRUE(t1.at(TensorIndices({0})) ==
              t1_broadcasted.at(TensorIndices({1, 1, 0})));

  EXPECT_TRUE(t1.at(TensorIndices({1})) ==
              t1_broadcasted.at(TensorIndices({0, 0, 1})));
  EXPECT_TRUE(t1.at(TensorIndices({1})) ==
              t1_broadcasted.at(TensorIndices({0, 1, 1})));
  EXPECT_TRUE(t1.at(TensorIndices({1})) ==
              t1_broadcasted.at(TensorIndices({1, 0, 1})));
  EXPECT_TRUE(t1.at(TensorIndices({1})) ==
              t1_broadcasted.at(TensorIndices({1, 1, 1})));

  EXPECT_TRUE(t1.at(TensorIndices({2})) ==
              t1_broadcasted.at(TensorIndices({0, 0, 2})));
  EXPECT_TRUE(t1.at(TensorIndices({2})) ==
              t1_broadcasted.at(TensorIndices({0, 1, 2})));
  EXPECT_TRUE(t1.at(TensorIndices({2})) ==
              t1_broadcasted.at(TensorIndices({1, 0, 2})));
  EXPECT_TRUE(t1.at(TensorIndices({2})) ==
              t1_broadcasted.at(TensorIndices({1, 1, 2})));

  EXPECT_TRUE(t2.at(TensorIndices({0, 0, 0})) ==
              t2_broadcasted.at(TensorIndices({0, 0, 0})));
  EXPECT_TRUE(t2.at(TensorIndices({0, 0, 0})) ==
              t2_broadcasted.at(TensorIndices({0, 0, 1})));
  EXPECT_TRUE(t2.at(TensorIndices({0, 0, 0})) ==
              t2_broadcasted.at(TensorIndices({0, 0, 2})));

  EXPECT_TRUE(t2.at(TensorIndices({0, 1, 0})) ==
              t2_broadcasted.at(TensorIndices({0, 1, 0})));
  EXPECT_TRUE(t2.at(TensorIndices({0, 1, 0})) ==
              t2_broadcasted.at(TensorIndices({0, 1, 1})));
  EXPECT_TRUE(t2.at(TensorIndices({0, 1, 0})) ==
              t2_broadcasted.at(TensorIndices({0, 1, 2})));

  EXPECT_TRUE(t2.at(TensorIndices({0, 2, 0})) ==
              t2_broadcasted.at(TensorIndices({0, 2, 0})));
  EXPECT_TRUE(t2.at(TensorIndices({0, 2, 0})) ==
              t2_broadcasted.at(TensorIndices({0, 2, 1})));
  EXPECT_TRUE(t2.at(TensorIndices({0, 2, 0})) ==
              t2_broadcasted.at(TensorIndices({0, 2, 2})));

  EXPECT_TRUE(t2.at(TensorIndices({1, 0, 0})) ==
              t2_broadcasted.at(TensorIndices({1, 0, 0})));
  EXPECT_TRUE(t2.at(TensorIndices({1, 0, 0})) ==
              t2_broadcasted.at(TensorIndices({1, 0, 1})));
  EXPECT_TRUE(t2.at(TensorIndices({1, 0, 0})) ==
              t2_broadcasted.at(TensorIndices({1, 0, 2})));

  EXPECT_TRUE(t2.at(TensorIndices({1, 1, 0})) ==
              t2_broadcasted.at(TensorIndices({1, 1, 0})));
  EXPECT_TRUE(t2.at(TensorIndices({1, 1, 0})) ==
              t2_broadcasted.at(TensorIndices({1, 1, 1})));
  EXPECT_TRUE(t2.at(TensorIndices({1, 1, 0})) ==
              t2_broadcasted.at(TensorIndices({1, 1, 2})));

  EXPECT_TRUE(t2.at(TensorIndices({1, 2, 0})) ==
              t2_broadcasted.at(TensorIndices({1, 2, 0})));
  EXPECT_TRUE(t2.at(TensorIndices({1, 2, 0})) ==
              t2_broadcasted.at(TensorIndices({1, 2, 1})));
  EXPECT_TRUE(t2.at(TensorIndices({1, 2, 0})) ==
              t2_broadcasted.at(TensorIndices({1, 2, 2})));
}