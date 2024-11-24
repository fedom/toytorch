#ifndef TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_LEAF_NODE_H__
#define TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_LEAF_NODE_H__
#include "nn/autograd/node.h"

namespace toytorch::autograd {

class LeafNodeBackward : public Node {
public:
  DEFINE_NODE_NAME_AND_ID(LeafNodeBackward)

  std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};
} // namespace toytorch::autograd

#endif // TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_LEAF_NODE_H__