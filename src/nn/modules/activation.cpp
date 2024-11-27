#include "nn/modules/activation.h"
#include "nn/operations/tensor_operations.h"
#include "nn/operations/activations.h"
#include "nn/autograd/autograd.h"
#include "nn/autograd/backward_node_activation_op.h"
#include "nn/exceptions/exceptions.h"
#include "nn/modules/activation_registry.h"
#include <iostream>

namespace toytorch {

// y = 1/(1 - e^(-x));
Tensor Sigmoid::forward(const Tensor& input) const {
  return sigmoid(input);
}

REGISTER_ACTIVATION(Sigmoid)

// y = (t + |t|) / 2;
Tensor Relu::forward(const Tensor& input) const {
  return relu(input);
}

REGISTER_ACTIVATION(Relu)

}  // namespace toytorch
