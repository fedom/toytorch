#include "toytorch.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// https://github.com/pytorch/examples/blob/main/cpp/regression/regression.cpp

#define POLY_DEGREE 4


// Builds features i.e. a matrix with columns [x, x^2, x^3, x^4].
toytorch::Tensor make_features(toytorch::Tensor x) {
  x = x.unsqueeze(1);
  std::vector<toytorch::Tensor> xs;
  for (int64_t i = 0; i < POLY_DEGREE; ++i)
    xs.push_back(x.pow(i + 1));
  return toytorch::cat(xs, 1);
}

// Approximated function.
toytorch::Tensor f(
    toytorch::Tensor x,
    toytorch::Tensor W_target,
    toytorch::Tensor b_target) {
  return matmul(x, W_target) + b_target;
}

// Creates a string description of a polynomial.
std::string poly_desc(toytorch::Tensor W, toytorch::Tensor b) {
  auto size = W.shape()[0];
  std::ostringstream stream;

  stream << "y = ";
  for (int64_t i = 0; i < size; ++i)
    stream << W[i] << " x^" << size - i << " ";
  stream << "+ " << b[0];
  return stream.str();
}

// Builds a batch i.e. (x, f(x)) pair.
std::pair<toytorch::Tensor, toytorch::Tensor> get_batch(
    toytorch::Tensor W_target,
    toytorch::Tensor b_target,
    int batch_size = 32) {
  auto random = toytorch::randn({batch_size});
  auto x = make_features(random);   // x : [32, 4]
  auto y = f(x, W_target, b_target);// 
  return std::make_pair(x, y);
}

int main() {
  // This are the real parameters that we need to prodict
  auto W_target = toytorch::randn({POLY_DEGREE, 1}) * 5;
  auto b_target = toytorch::randn({1}) * 5;

  // Define a linear layer to approximate the W_target and b_target
  auto fc = toytorch::Linear(W_target.shape()[0], 1);
  toytorch::optim::SGD optim(fc.parameters(), .1);

  float loss = 0;
  int64_t batch_idx = 0;

  while (++batch_idx) {
    // Get data
    toytorch::Tensor batch_x, batch_y;
    std::tie(batch_x, batch_y) = get_batch(W_target, b_target);

    // Reset gradients
    optim.zero_grad();

    // Forward pass
    auto loss_tensor = toytorch::smooth_l1_loss(fc.forward(batch_x), batch_y);
    loss = loss_tensor[0];

    // Backward pass
    loss_tensor.backward();

    // Apply gradients
    optim.step();

    // std::cout << "loss:" << loss << std::endl;

    // Stop criterion
    if (loss < 1e-3f)
      break;
  }

  std::cout << "Loss: " << loss << " after " << batch_idx << " batches"
            << std::endl;
            
  std::cout << "==> Learned function:\t"
            << poly_desc(fc.weights().view({-1}), fc.bias()) << std::endl;
  std::cout << "==> Actual function:\t"
            << poly_desc(W_target.view({-1}), b_target) << std::endl;

  return 0;
}