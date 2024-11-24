#ifndef TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_UNIMPLEMENT_OP_H__
#define TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_UNIMPLEMENT_OP_H__
#include "nn/tensor/tensor.h"
#include "nn/autograd/node.h"
#include <string>

namespace toytorch::autograd {

class UnimplementNodeBackward : public Node {
public:
  UnimplementNodeBackward(const std::string &msg) : msg_(msg) {}
  std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;

private:
  std::string msg_;
};

} // namespace toytorch::autograd

#endif // TOYTORCH_NN_AUTOGRAD_BACKWARD_NODE_UNIMPLEMENT_OP_H__