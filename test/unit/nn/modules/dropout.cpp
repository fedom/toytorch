#include "nn/modules/dropout.h"
#include "nn/operations/tensor_operations.h"
#include "nn/tensor/tensor_creator.h"
#include <gtest/gtest.h>

using namespace toytorch;

TEST(DropoutTest, dropout) {
  Dropout dropout_layer(0.4);

  dropout_layer.train();

  Tensor input = ones({4, 5});
  Tensor output = dropout_layer.forward(input);

  output.print_shape();
  output.print();

}

TEST(DropoutTest, dropout2d) {
  Dropout2d dropout_layer(0.2);

  dropout_layer.train();

  Tensor input = ones({4, 5, 3, 2});
  Tensor output = dropout_layer.forward(input);

  output.print_shape();
  output.print();
}