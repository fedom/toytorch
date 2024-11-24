#ifndef TOYTORCH_NN_AUTOGRAD_TENSOR_GRAD_INFO_H__
#define TOYTORCH_NN_AUTOGRAD_TENSOR_GRAD_INFO_H__
#include <memory>

namespace toytorch {

class Tensor;
namespace autograd {

class Node;
struct GradInfo {
  std::shared_ptr<Tensor> grad;
  std::shared_ptr<Node> grad_fn;
};
}  // namespace autograd

}  // namespace toytorch

#endif  // TOYTORCH_NN_AUTOGRAD_TENSOR_GRAD_INFO_H__