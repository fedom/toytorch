#include "nn/autograd/backward_node_leaf_op.h"

namespace toytorch::autograd {
  
std::vector<Tensor> LeafNodeBackward::apply(std::vector<Tensor>&& grads) {
  return std::vector<Tensor>();
}
}  // namespace toytorch::autograd
