#include "toytorch.h"
#include <gtest/gtest.h>

using namespace toytorch;

TEST(DebugUtilsTest, PrintGraph) {
  Linear fc1(4, 32, "Sigmoid", "fc1");
  Linear fc2(32, 8, "Relu", "fc2");
  Tensor custom_param = randn({4, 8}, true);

  Tensor input = rand({5, 4});

  Tensor output1 = fc1.forward(input); // (5, 32)
  output1 = fc2.forward(output1);       // (5, 8)

  Tensor output2 = matmul(input, custom_param); // (5, 8)
  Tensor result = (output1 * output2).sum();

  // std::cout << debug::print_backward_graph(result) << std::endl;
  result.backward();

  // Print the gradient of each parameter after backward()
  // fc1.weights().grad()->print();
  // fc1.bias().grad()->print();
  // fc2.weights().grad()->print();
  // fc2.bias().grad()->print();
  // custom_param.grad()->print();
}