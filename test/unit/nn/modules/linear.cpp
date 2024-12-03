#include "nn/modules/linear.h"
#include "nn/tensor/tensor_creator.h"
#include <gtest/gtest.h>

using namespace toytorch;

TEST(LinearTest, forward) {
  nn::Linear model(64, 16);
  Tensor input = randn({10, 64});
  Tensor result = model.forward(input);

  EXPECT_TRUE(result.shape() == TensorShape({10, 16}));

  // std::cout << "=====model params:=====\n";
  // model.weights().print();
  // model.bias().print();

  // std::cout << "=====input tensor:=====\n";
  // input.print();

  // std::cout << "=====result:=====\n";
  // result.print();
}

// TEST(LinearTest, forward) {
//   Linear model(3, 1);
//   Tensor input = randn({4, 3});
//   Tensor result = model.forward(input);

//   // result is (10, 16)


//   EXPECT_TRUE(result.shape() == TensorShape({4, 1}));

//   // std::cout << "=====model params:=====\n";
//   // model.weights().print();
//   // model.bias().print();

//   // std::cout << "=====input tensor:=====\n";
//   // input.print();

//   // std::cout << "=====result:=====\n";
//   // result.print();
// }