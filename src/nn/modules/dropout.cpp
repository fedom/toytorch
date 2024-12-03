#include "nn/modules/dropout.h"
#include "nn/operations/dropout.h"

namespace toytorch::nn {

Tensor Dropout::forward(const Tensor& input) const {
  // training is set by user before start training bying calling model.train()
  return dropout(input, p_, training_);
}


Tensor Dropout2d::forward(const Tensor& input) const {
  // training is set by user before start training bying calling model.train()
  return dropout2d(input, p_, training_);
}

} // namespace
