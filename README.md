# ToyTorch
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This project was created as a practice exercise while I was learning about neural network with PyTorch. To deepen my understanding, I wrote some test code to mimic PyTorch's behavior. Over time, this evolved into simple neural network engine. I made it available for anyone who wants to quickly grasp the concepts by experimenting with it. 

Please note that efficiency and elegance were not prioritized. The main goal is ease of understanding. Treat it as a fun experiment and enjoy!

## **Features**
To keep it simple, I targeted only the `float` data type and `cpu` device. (Pytorch supports different data types and devices through a dispatcher mechanism which I haven't tried yet). 

Current features include:
- **A Tensor class**
- **Tensor creators**
  - empty, zero, ones, rand, randn
    
- **Basic Tensor operations**
  - matmul
  - add, sub, mul, div, pow, exp, neg, abs, sign, sum, mean
  - where, select, transpose, cat, squeeze, unsqueeze, view
  - gt, ge, lt, le, strict_equal, strict_allclose
    
- **Modules**
  - Linear
    
- **Activation function**
  - Sigmoid, Relu
    
- **Autograd mechanism**
  - Dynamic update the backward graph if the tensor requires_grad=True
  - Support basic operations matmul, add, sub, mul, div, pow, where, view
    
- **Optimizer**
  - SGD
    
- **Debug**
  - Output backward graph in DOT format

## **Build**
Since my environment is c++20, I used `Concept` in a random number util template. But it should be very simple to replace it if you want to build with prior version. Building unit tests need `GTest`. You can skip it by addding `-DBUILD_TESTS=OFF`.

Enter the root fold and run
```
cmake -B build    // Add -DBUILD_TESTS=OFF if you want to skip building the unit tests
cmake --build build
```
Two binaries will be created:
```
./build/src/linear_regression  // linear regression example provided by pytorch implemented using our toy
./build/test/nn_example_test   // unittests if you have built it
```
Run them to check the results.

## **Example**

```c++
#include "toytorch.h"

int main() {
  Linear fc1(4, 32, "Sigmoid", "fc1");
  Linear fc2(32, 8, "Relu", "fc2");
  Linear fc3(8, 1, "Sigmoid", "fc3");

  Tensor input = rand({5, 4});
  Tensor result = fc1.forward(input);
  result = fc2.forward(result);
  result = fc3.forward(result);
  result = result.sum();

  std::cout << debug::print_backward_graph(result) << std::endl;
  result.backward();

  // Print the gradient of each parameter after backward()
  fc1.weights().grad()->print();
  fc1.bias().grad()->print();
  fc2.weights().grad()->print();
  fc2.bias().grad()->print();
  fc3.weights().grad()->print();
  fc3.bias().grad()->print();

  return 0;
}
```
Here's the backward graph printed.
![backward graph](docs/images/backward.svg)

## **What to do next**

As you can see, we only have a skeleton here. We are missing a dataloader, a model saver, and most of the necessary operations required to support even basic use cases. After becoming familiar with it, you can try adding more features as you like. You can aim to support the [MNIST](https://github.com/pytorch/examples/blob/main/cpp/mnist/mnist.cpp) example as a starting point.

## **Resources**

Besides the official documents, thanks for the amazing blogs from ezyang. I've learned a lot from it. 

- [ezyang’s blog](http://blog.ezyang.com/category/pytorch/)
- [PyTorch resources](https://github.com/pytorch/pytorch/tree/main?tab=readme-ov-file#getting-started)

## **Communication**
[Filing an issue](https://github.com/fedom/ToyTorch/issues)

## **License**
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.