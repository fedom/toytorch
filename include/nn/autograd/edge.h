#ifndef TOYTORCH_NN_AUTOGRAD_EDGE_H__
#define TOYTORCH_NN_AUTOGRAD_EDGE_H__
#include "nn/tensor/tensor.h"

namespace toytorch::autograd {

class Node;
class Edge {
 public:
  Edge(const Tensor& tensor) : tensor_(tensor) {}

  Tensor& get_tensor() { return tensor_; }

  // Edge can point to null node if the edge's tensor doesn't
  // require grad. In that case, it is only used as constant
  // in grad calculation for other variables
  std::shared_ptr<Node> get_node() {
    if (tensor_.grad_info()) {
      return tensor_.grad_info()->grad_fn;
    }
    return nullptr;
  }

 private:
  Tensor tensor_;
};
}  // namespace toytorch::autograd

#endif  // TOYTORCH_NN_AUTOGRAD_EDGE_H__