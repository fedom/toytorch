#include "nn/modules/linear.h"
#include <iostream>
#include <memory>
#include "nn/exceptions/exceptions.h"
#include "nn/operations/matrix.h"
#include "nn/operations/tensor_operations.h"

namespace toytorch::nn {

Linear::Linear(int num_input, int num_output, const std::string& act_name,
               const std::string& name)
    : activation_(GET_ACTIVATION(act_name)), name_(name) {
  UniformRandomGenerator rng(-1.0f / num_input, 1.0f / num_input);

  weights_ = Tensor(TensorShape({num_input, num_output}), rng, true);
  register_parameter("W", weights_);

  bias_ = Tensor(TensorShape({1, num_output}), rng, true);

  register_parameter("B", bias_);
}

Tensor Linear::forward(const Tensor& input) const {

  Tensor out = matmul(input, weights_);
  out = out + (bias_);
  if (activation_) {
    out = activation_->forward(out);
  }

  return out;
}

}  // namespace toytorch::nn