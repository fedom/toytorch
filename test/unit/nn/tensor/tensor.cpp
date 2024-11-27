#include "nn/tensor/tensor.h"
#include "nn/tensor/tensor_creator.h"
#include "nn/operations/tensor_helper.h"
#include "nn/operations/tensor_operations.h"
#include "nn/exceptions/exceptions.h"
#include <gtest/gtest.h>

using namespace toytorch;

TEST(TensorTest, CreateScalar) {
  Tensor tensor(3);

  EXPECT_TRUE(tensor.is_scalar());
  EXPECT_TRUE(tensor.dim() == 0);
  EXPECT_TRUE(tensor.shape().size() == 0);
  EXPECT_TRUE(tensor[0] == 3);
}

TEST(TensorTest, CreateWithDim) {
  Tensor tensor({3, 4});

  EXPECT_FALSE(tensor.is_scalar());
  EXPECT_TRUE(tensor.dim() == 2);
  EXPECT_TRUE(tensor.shape() == TensorShape({3, 4}));
}

TEST(TensorTest, CreateWithRandGenerator) {
  UniformRandomGenerator gen1(1.0f, 2.0f);
  NormalRandomGenerator gen2(0.0f, 1.0f);
  Tensor t1({3, 4}, gen1);
  Tensor t2({3, 4}, gen2);

  t1.print();
  std::cout << "==\n";
  t2.print();

  // EXPECT_FALSE(tensor.is_scalar());
  // EXPECT_TRUE(tensor.dim() == 2);
  // EXPECT_TRUE(tensor.shape() == TensorShape({3, 4}));
}

TEST(TensorTest, FillTensor) {
  Tensor t1({3, 2});
  std::vector<float> val = {1, 2, 3, 4, 5, 6};

  EXPECT_THROW(t1.fill({1, 2, 3, 4, 5}), ExceptionInvalidArgument);
  EXPECT_NO_THROW(t1.fill(val));
  EXPECT_TRUE(memcmp(t1.raw_data(), val.data(), val.size()) == 0);

  Tensor t2({2, 3});
  EXPECT_NO_THROW(t2.fill(val));
  EXPECT_TRUE(memcmp(t1.raw_data(), val.data(), val.size()) == 0);

  Tensor t3 = t2.transpose();
  EXPECT_THROW(t3.fill(val), ExceptionNotImpl);
}

TEST(TensorTest, CompareTensorDiffShape) {
  Tensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t2({2, 3}, {1, 2, 3, 4, 5, 6});

  EXPECT_FALSE(t1 == t2);
  EXPECT_TRUE(t1 != t2);
}

TEST(TensorTest, SharedCopy) {
  Tensor t1({2, 3});
  Tensor t2(t1);

  EXPECT_TRUE(t1 == t2);
  t1[0] = 2;
  EXPECT_TRUE(t1 == t2);
}

TEST(TensorTest, DeepCopy) {
  Tensor t1({2, 3}, {1, 2, 3, 4, 5, 6});

  Tensor t2 = t1.deep_copy();
  EXPECT_TRUE(t1 == t2);

  t1[0] = 2;
  EXPECT_TRUE(t1 != t2);

  t2[0] = 2;
  EXPECT_TRUE(t1 == t2);
}


TEST(TensorTest, TensorTranspose) {
  Tensor t1({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor t2 = t1.transpose();

  Tensor expected_result = Tensor({2, 3}, {1, 3, 5, 2, 4, 6});
  EXPECT_TRUE(t2 == expected_result);
}


TEST(TensorTest, SelectKeepDim) {
  Tensor t({2, 3}, {1,2,3,4,5,6});

  EXPECT_TRUE(t.select(0, 0, true) == Tensor({1, 3}, {1, 2, 3}));
  EXPECT_TRUE(t.select(0, 1, true) == Tensor({1, 3}, {4, 5, 6}));

  EXPECT_TRUE(t.select(1, 0, true) == Tensor({2, 1}, {1, 4}));
  EXPECT_TRUE(t.select(1, 1, true) == Tensor({2, 1}, {2, 5}));
  EXPECT_TRUE(t.select(1, 2, true) == Tensor({2, 1}, {3, 6}));
}

TEST(TensorTest, SelectNotKeepDim) {
  Tensor t({2, 3}, {1,2,3,4,5,6});

  EXPECT_TRUE(t.select(0, 0) == Tensor({3}, {1, 2, 3}));
  EXPECT_TRUE(t.select(0, 1) == Tensor({3}, {4, 5, 6}));

  EXPECT_TRUE(t.select(1, 0) == Tensor({2}, {1, 4}));
  EXPECT_TRUE(t.select(1, 1) == Tensor({2}, {2, 5}));
  EXPECT_TRUE(t.select(1, 2) == Tensor({2}, {3, 6}));
}

TEST(TensorTest, SelectNotKeepDimScalar) {
  Tensor t({6}, {1,2,3,4,5,6});

  EXPECT_TRUE(t.select(0, 0) == Tensor(1));
  EXPECT_TRUE(t.select(0, 1) == Tensor(2));

  EXPECT_TRUE(t.select(0, 0).is_scalar());
}

TEST(TensorTest, SumKeepDim) {
  Tensor t({2, 3}, {1,2,3,4,5,6});

  EXPECT_TRUE(t.sum(0, true) == Tensor({1, 3}, {5, 7, 9}));
  EXPECT_TRUE(t.sum(1, true) == Tensor({2, 1}, {6, 15}));
}

TEST(TensorTest, SumNotKeepDim) {
  Tensor t({2, 3}, {1,2,3,4,5,6});

  EXPECT_TRUE(t.sum(0) == Tensor({3}, {5, 7, 9}));
  EXPECT_TRUE(t.sum(1) == Tensor({2}, {6, 15}));
}

TEST(TensorTest, SumNotKeepDimScalar) {
  Tensor t({6}, {1,2,3,4,5,6});

  Tensor a = t.sum(0);
  EXPECT_TRUE(t.sum(0) == Tensor(21));
  EXPECT_TRUE(t.sum(0).is_scalar());
}

TEST(TensorTest, SumMultiAxesKeepDim) {
  Tensor t({2, 3}, {1,2,3,4,5,6});

  EXPECT_TRUE(t.sum({0, 1}, true) == Tensor({1, 1}, std::vector<float>({21})));
  EXPECT_THROW(t.sum({1, 0}, true), ExceptionInvalidArgument);
}

TEST(TensorTest, SumMultiAxesNotKeepDim) {
  Tensor t({2, 3}, {1,2,3,4,5,6});

  // Should be a scalar
  EXPECT_TRUE(t.sum({0, 1}, false) == Tensor(21));
  EXPECT_THROW(t.sum({1, 0}, false), ExceptionInvalidArgument);
}

TEST(TensorTest, Expand) {
  Tensor t1 = Tensor({2, 3}, {1,2,3,4,5,6});

  t1 = t1.unsqueeze(0);
  ASSERT_TRUE(t1.shape() == TensorShape({1, 2, 3}));

  Tensor t2 = t1.expand({3,2,3});
  EXPECT_TRUE(t2.shape() == TensorShape({3, 2, 3}));
  EXPECT_TRUE(t2.strides() == TensorShape({0, 3, 1}));
}

TEST(TensorTest, IsContiguous) {
  Tensor t1 = ones({2, 3, 3});
  EXPECT_TRUE(t1.is_contiguous());

  Tensor t2 = t1.transpose(0, 1);
  ASSERT_TRUE(t2.shape() == TensorShape({3,2,3}));
  EXPECT_FALSE(t2.is_contiguous());

  Tensor t3 = ones({1, 3, 3});
  Tensor t4 = t3.expand({4, 3, 3});

  ASSERT_TRUE(t4.shape() == TensorShape({4,3,3}));
  EXPECT_FALSE(t4.is_contiguous());
}

TEST(TensorTest, View) {
  Tensor t1 = ones({2, 3, 3, 4, 5});

  Tensor t = t1.view({-1});
  EXPECT_TRUE(t.shape() == TensorShape({360}));
  EXPECT_TRUE(t.view({-1}).strides() == TensorShape({1}));

  t = t1.view({-1, 3});
  EXPECT_TRUE(t.shape() == TensorShape({120, 3}));
  EXPECT_TRUE(t.strides() == TensorShape({3, 1}));

  t = t1.view({2, -1, 5});
  EXPECT_TRUE(t.shape() == TensorShape({2, 36, 5}));
  EXPECT_TRUE(t.strides() == TensorShape({180 ,5, 1}));


  t = t1.view({2, 5, -1});
  EXPECT_TRUE(t.shape() == TensorShape({2, 5, 36}));
  EXPECT_TRUE(t.strides() == TensorShape({180 ,36, 1}));

  // can't have more than one -1
  EXPECT_THROW(t1.view({-1, 3, -1, 5}), ExceptionInvalidArgument);

  // Not divisible
  EXPECT_THROW(t1.view({7, -1}), ExceptionTensorShapeIncompatible);

  Tensor t2 = t1.transpose(0, 1);
  Tensor t3 = t1.unsqueeze(1);
  t3 = t3.expand({2, 2, 3, 3, 4, 5});

  // uncontiguous tensor not supported
  ASSERT_TRUE(t2.shape() == TensorShape({3, 2, 3, 4, 5}));
  ASSERT_TRUE(t3.shape() == TensorShape({2, 2, 3, 3, 4, 5}));
  EXPECT_THROW(t2.view({-1}), ExceptionNotImpl);
  EXPECT_THROW(t3.view({-1}), ExceptionNotImpl);

  Tensor t4 = ones({128});
  EXPECT_TRUE(t4.view({2, 4, 8, 2}) == ones({2, 4, 8, 2}));
  EXPECT_THROW(t4.view({32, 4, 2}), ExceptionTensorShapeIncompatible);
}

TEST(TensorTest, add_) {
  Tensor t1 = ones({2, 3});
  Tensor t2 = ones({3});

  EXPECT_THROW(t2.add_(t1), ExceptionTensorShapeIncompatible);
  EXPECT_NO_THROW(t1.add_(t2));
  EXPECT_TRUE(t1 == ones({2,3}) * 2);

  Tensor t3 = Tensor(1);
  Tensor t4 = Tensor(2);
  
  EXPECT_NO_THROW(t3.add_(t4));
  EXPECT_TRUE(t3 == Tensor(3));

  Tensor t5 = ones({2, 3});
  Tensor t6 = Tensor(1);

  EXPECT_NO_THROW(t5.add_(t6));
  EXPECT_TRUE(t5 == ones({2, 3}) * 2);
}

TEST(TensorTest, sub_) {
  Tensor t1 = ones({2, 3}) * 2;
  Tensor t2 = ones({3});

  EXPECT_THROW(t2.sub_(t1), ExceptionTensorShapeIncompatible);
  EXPECT_NO_THROW(t1.sub_(t2));
  EXPECT_TRUE(t1 == ones({2,3}));
}

TEST(TensorTest, mul_) {
  Tensor t1 = ones({2, 3}) * 2;
  Tensor t2 = ones({3}) * 2;

  EXPECT_THROW(t2.mul_(t1), ExceptionTensorShapeIncompatible);
  EXPECT_NO_THROW(t1.mul_(t2));
  EXPECT_TRUE(t1 == ones({2,3}) * 4);
}

TEST(TensorTest, div_) {
  Tensor t1 = ones({2, 3}) * 4;
  Tensor t2 = ones({3}) * 2;

  EXPECT_THROW(t2.div_(t1), ExceptionTensorShapeIncompatible);
  EXPECT_NO_THROW(t1.div_(t2));
  EXPECT_TRUE(t1 == ones({2,3}) * 2);
}