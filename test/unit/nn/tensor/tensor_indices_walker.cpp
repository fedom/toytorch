#include "nn/tensor/tensor_indices_walker.h"
#include <gtest/gtest.h>
#include "nn/tensor/tensor.h"
#include "nn/tensor/types.h"
#include "toytorch.h"

using namespace toytorch;
TEST(TensorIndicesWalkerTest, WalkNormal) {

  Tensor t1 = arange(0, 12).view({2, 2, 3});
  TensorIndices indices = t1.get_indices();
  TensorIndicesWalker walker(t1.shape(), indices);

  std::vector<int> v;
  do {
    v.push_back(t1.at(indices));
  } while (walker.step());

  EXPECT_TRUE(indices == TensorIndices({0, 0, 0}));
  EXPECT_TRUE(v == std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
}

TEST(TensorIndicesWalkerTest, narrow_to_index) {

  Tensor t1 = arange(0, 24).view({2, 3, 4});

  TensorIndices indices = t1.get_indices();
  TensorIndicesWalker walker(t1.shape(), indices);
  walker.narrow_to_index(1, 1);

  std::vector<int> v;
  do {
    v.push_back(t1.at(indices));
  } while (walker.step());

  EXPECT_TRUE(indices == TensorIndices({0, 1, 0}));
  EXPECT_TRUE(v == std::vector<int>({4, 5, 6, 7, 16, 17, 18, 19}));

  walker.reset();
  v.clear();

  do {
    v.push_back(t1.at(indices));
  } while (walker.step());

  EXPECT_TRUE(indices == TensorIndices({0, 0, 0}));
  EXPECT_TRUE(
      v == std::vector<int>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}));


  walker.reset();
  walker.narrow_to_index(-1, 0);

  v.clear();
  do {
    v.push_back(t1.at(indices));
  } while (walker.step());

  EXPECT_TRUE(indices == TensorIndices({0, 0, 0}));
  EXPECT_TRUE(v == std::vector<int>({0, 4, 8, 12, 16, 20}));
}

TEST(TensorIndicesWalkerTest, narrow_to_index_range) {

  Tensor t1 = arange(0, 24).view({2, 3, 4});

  TensorIndices indices = t1.get_indices();
  TensorIndicesWalker walker(t1.shape(), indices);
  walker.narrow_to_index_range(1, 0, 2);

  std::vector<int> v;
  do {
    v.push_back(t1.at(indices));
  } while (walker.step());

  EXPECT_TRUE(indices == TensorIndices({0, 0, 0}));
  EXPECT_TRUE(v == std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16,
                                     17, 18, 19}));

  walker.reset();
  v.clear();

  do {
    v.push_back(t1.at(indices));
  } while (walker.step());

  EXPECT_TRUE(
      v == std::vector<int>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}));

  walker.reset();
  walker.narrow_to_index_range(-1, 1, 3);
  v.clear();
  do {
    v.push_back(t1.at(indices));
  } while (walker.step());

  EXPECT_TRUE(indices == TensorIndices({0, 0, 1}));
  EXPECT_TRUE(v == std::vector<int>({1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15,
                                     17, 18, 19, 21, 22, 23}));

}


TEST(TensorIndicesWalkerTest, freeze_dim_range) {

  Tensor t1 = arange(0, 24).view({2, 3, 4});

  TensorIndices indices = t1.get_indices();
  TensorIndicesWalker walker(t1.shape(), indices);
  walker.freeze_dim_range(0, 1);

  std::vector<int> v;
  do {
    v.push_back(t1.at(indices));
  } while (walker.step());

  EXPECT_TRUE(indices == TensorIndices({0, 0, 0}));
  EXPECT_TRUE(v == std::vector<int>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11}));

  walker.reset();
  walker.freeze_dim_range(1, 2);
  v.clear();
  do {
    v.push_back(t1.at(indices));
  } while (walker.step());

  EXPECT_TRUE(indices == TensorIndices({0, 0, 0}));
  EXPECT_TRUE(v == std::vector<int>({0, 12}));

}

TEST(TensorIndicesWalkerTest, set_dim_stride) {

  Tensor t1 = arange(0, 24).view({2, 3, 4});

  TensorIndices indices = t1.get_indices();
  TensorIndicesWalker walker(t1.shape(), indices);
  walker.set_dim_stride(-1, 2);

  std::vector<int> v;
  do {
    v.push_back(t1.at(indices));
  } while (walker.step());

  EXPECT_TRUE(indices == TensorIndices({0, 0, 0}));
  EXPECT_TRUE(v == std::vector<int>({0, 2, 4, 6,  8, 10, 12, 14, 16, 18, 20, 22}));

  walker.reset();
  walker.set_dim_stride(1, 3);
  v.clear();
  do {
    v.push_back(t1.at(indices));
  } while (walker.step());

  EXPECT_TRUE(v == std::vector<int>({0, 1, 2, 3, 12, 13, 14, 15}));
}

TEST(TensorIndicesWalkerTest, reset) {

  Tensor t1 = arange(0, 24).view({2, 3, 4});

  TensorIndices indices = t1.get_indices();
  TensorIndicesWalker walker(t1.shape(), indices);
  walker.set_dim_stride(-1, 2);

  std::vector<int> v;
  do {
    v.push_back(t1.at(indices));
  } while (walker.step());

  EXPECT_TRUE(indices == TensorIndices({0, 0, 0}));
  EXPECT_TRUE(v == std::vector<int>({0, 2, 4, 6,  8, 10, 12, 14, 16, 18, 20, 22}));

  walker.set_dim_stride(1, 2);
  v.clear();
  do {
    v.push_back(t1.at(indices));
  } while (walker.step());

  EXPECT_TRUE(v == std::vector<int>({0, 2, 8, 10, 12, 14, 20, 22}));
}

TEST(TensorIndicesWalkerTest, Mixed) {

  Tensor t1 = arange(0, 24).view({2, 3, 4});

  TensorIndices indices = t1.get_indices();
  TensorIndicesWalker walker(t1.shape(), indices);

  walker.set_dim_stride(-1, 2);
  walker.freeze_dim_range(0, 1);
  walker.narrow_to_index(1, 2);

  std::vector<int> v;
  do {
    v.push_back(t1.at(indices));
  } while (walker.step());

  EXPECT_TRUE(indices == TensorIndices({0, 2, 0}));
  EXPECT_TRUE(v == std::vector<int>({8, 10}));

  walker.reset();
  v.clear();
  do {
    v.push_back(t1.at(indices));
  } while (walker.step());

  EXPECT_TRUE(indices == TensorIndices({0, 0, 0}));
  EXPECT_TRUE(
      v == std::vector<int>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}));
}