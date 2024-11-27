#include "nn/autograd/backward_node_unimplement_op.h"
#include "nn/exceptions/exceptions.h"

namespace toytorch::autograd {


std::vector<Tensor> UnimplementNodeBackward::apply(
    std::vector<Tensor>&& grads) {
      throw ExceptionNotImpl(msg_);
    }

}  // namespace toytorch::autograd