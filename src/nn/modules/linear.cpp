#include "nn/modules/linear.h"
#include <iostream>
#include <memory>
#include "exception/exceptions.h"
#include "nn/tensor/tensor_operations.h"

namespace toytorch {

Linear::Linear(int num_input, int num_output,
                         const std::string& act_name, const std::string &name)
    : activation_(ActivationRegistry::instance().get(act_name)), name_(name) {
  UniformRandomGenerator rng(-1.0f / num_input, 1.0f / num_input);

  weights_ = register_parameter(
      "W", Tensor(TensorShape({num_input, num_output}), rng, true));
  bias_ =
      register_parameter("B", Tensor(TensorShape({1, num_output}), rng, true));
}

Tensor Linear::forward(const Tensor& input) const {

  Tensor out = matmul(input, weights_);
  out = out + (bias_);
  if (activation_) {
    out = activation_->forward(out);
  }

  return out;
}

}  // namespace toytorch