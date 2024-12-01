#include "nn/utils/extended_vector.h"
#include <gtest/gtest.h>
#include "nn/exceptions/exceptions.h"

using namespace toytorch;

TEST(ExtendedVectorTest, Equality) {
  ExtendedVector<int> v1{2, 3, 4};
  ExtendedVector<int> v2{6, 7, 8};

  EXPECT_TRUE(ExtendedVector<int>({1, 2, 3}) == ExtendedVector<int>({1, 2, 3}));
  EXPECT_TRUE(ExtendedVector<int>({1, 2, 3}) != ExtendedVector<int>({1, 3}));
  EXPECT_TRUE(ExtendedVector<int>({1, 2, 3}) != ExtendedVector<int>({1, 3, 2}));
}

TEST(ExtendedVectorTest, At) {
  ExtendedVector<int> v1{2, 3, 4};

  EXPECT_TRUE(v1.at(0) == 2);
  EXPECT_TRUE(v1.at(1) == 3);
  EXPECT_TRUE(v1.at(2) == 4);
  EXPECT_TRUE(v1.at(-1) == 4);
  EXPECT_TRUE(v1.at(-2) == 3);
  EXPECT_TRUE(v1.at(-3) == 2);

  EXPECT_THROW(v1.at(-4), std::out_of_range);
  EXPECT_THROW(v1.at(4), std::out_of_range);
}

TEST(ExtendedVectorTest, Subvector) {
  ExtendedVector<int> v1{1, 2, 3, 4, 5, 6};

  EXPECT_TRUE(v1.subvector(1) == ExtendedVector<int>({2, 3, 4, 5, 6}));
  EXPECT_TRUE(v1.subvector(1, 2) == ExtendedVector<int>({2, 3}));
  EXPECT_TRUE(v1.subvector(0) == v1);
  EXPECT_TRUE(v1.subvector(0, 6) == v1);
  EXPECT_THROW(v1.subvector(0, 7), ExceptionInvalidArgument);
  EXPECT_THROW(v1.subvector(1, 6), ExceptionInvalidArgument);
}

TEST(ExtendedVectorTest, expand_left_to) {
  ExtendedVector<int> v1{1, 2, 3};

  v1.expand_left_to(6, 6);
  EXPECT_TRUE(v1 == ExtendedVector<int>({6, 6, 6, 1, 2, 3}));
}

TEST(ExtendedVectorTest, expand_left_to_copy) {
  ExtendedVector<int> v1{1, 2, 3};

  ExtendedVector<int> v2 = v1.expand_left_to_copy(5, 5);
  EXPECT_TRUE(v2 == ExtendedVector<int>({5, 5, 1, 2, 3}));
  EXPECT_TRUE(v1 == ExtendedVector<int>({1, 2, 3}));
}

TEST(ExtendedVectorTest, insert) {
  ExtendedVector<int> v1{1, 2, 3};

  // clang-format off
  v1.insert(2, 5);
  EXPECT_TRUE(v1 == ExtendedVector<int>({1, 2, 5, 3}));

  v1.insert(0, 6);
  EXPECT_TRUE(v1 == ExtendedVector<int>({6, 1, 2, 5, 3}));

  v1.insert(-1, 7);
  EXPECT_TRUE(v1 == ExtendedVector<int>({6, 1, 2, 5, 7, 3}));

  v1.insert(-2, 8);
  EXPECT_TRUE(v1 == ExtendedVector<int>({6, 1, 2, 5, 8, 7, 3}));
}

TEST(ExtendedVectorTest, insert_copy) {
  ExtendedVector<int> v1{1, 2, 3};

  auto v2 = v1.insert_copy(2, 5);
  EXPECT_TRUE(v2 == ExtendedVector<int>({1, 2, 5, 3}));
  EXPECT_TRUE(v1 == ExtendedVector<int>({1, 2, 3}));
}

TEST(ExtendedVectorTest, remove) {
  ExtendedVector<int> v1{1, 2, 3, 4, 5, 6, 7};

  v1.remove(-1);
  EXPECT_TRUE(v1 == ExtendedVector<int>({1, 2, 3, 4, 5, 6}));  

  v1.remove(0);
  EXPECT_TRUE(v1 == ExtendedVector<int>({2, 3, 4, 5, 6})); 

  v1.remove(-3);
  EXPECT_TRUE(v1 == ExtendedVector<int>({2, 3, 5, 6})); 
}


TEST(ExtendedVectorTest, remove_copy) {
  ExtendedVector<int> v1{1, 2, 3, 4, 5, 6, 7};

  auto v2 = v1.remove_copy(0);

  // clang-format off
  EXPECT_TRUE(v2 == ExtendedVector<int>({2, 3, 4, 5, 6, 7})); 
  EXPECT_TRUE(v1 == ExtendedVector<int>({1, 2, 3, 4, 5, 6, 7})); 

}

TEST(ExtendedVectorTest, split2) {
  ExtendedVector<int> v1{1, 2, 3, 4, 5, 6, 7};

  auto &&[v2, v3] = v1.split2(-2);
  EXPECT_TRUE(v2 == ExtendedVector<int>({1, 2, 3, 4, 5}));
  EXPECT_TRUE(v3 == ExtendedVector<int>({6, 7}));

  auto &&[v4, v5] = v1.split2(0);
  EXPECT_TRUE(v4.empty());
  EXPECT_TRUE(v5 == v1);

  auto &&[v6, v7] = v1.split2(-1);
  EXPECT_TRUE(v6 == ExtendedVector<int>({1, 2, 3, 4, 5, 6}));
  EXPECT_TRUE(v7 ==  ExtendedVector<int>({7}));
}

TEST(ExtendedVectorTest, split3) {

  ExtendedVector<int> v1{1, 2, 3, 4, 5, 6, 7};

  auto &&[v2, v3, v4] = v1.split3(0);

  EXPECT_TRUE(v2.empty());
  EXPECT_TRUE(v3 == ExtendedVector<int>({1}));
  EXPECT_TRUE(v4 == ExtendedVector<int>({2, 3, 4, 5, 6, 7}));

  auto &&[v5, v6, v7] = v1.split3(3);

  EXPECT_TRUE(v5 == ExtendedVector<int>({1, 2, 3}));
  EXPECT_TRUE(v6 == ExtendedVector<int>({4}));
  EXPECT_TRUE(v7 == ExtendedVector<int>({5, 6, 7}));


  auto &&[v8, v9, v10] = v1.split3(-1);
  EXPECT_TRUE(v8 == ExtendedVector<int>({1, 2, 3, 4, 5, 6}));
  EXPECT_TRUE(v9 == ExtendedVector<int>({7}));
  EXPECT_TRUE(v10.empty());
}

TEST(ExtendedVectorTest, concat) {
  ExtendedVector<int> v1{1, 2};
  ExtendedVector<int> v2{5, 6, 7};

  v1.concat(v2);
  EXPECT_TRUE(v1 == ExtendedVector<int>({1, 2, 5, 6, 7}));
}

TEST(ExtendedVectorTest, concat_copy) {
  ExtendedVector<int> v1{1, 2};
  ExtendedVector<int> v2{5, 6, 7};

  auto v3 = v1.concat_copy(v2);
  EXPECT_TRUE(v3 == ExtendedVector<int>({1, 2, 5, 6, 7}));
  EXPECT_TRUE(v1 == ExtendedVector<int>({1, 2}));
}
