
#ifndef TOYTORCH_NN_AUTOGRAD_AUTOGRAD_H__
#define TOYTORCH_NN_AUTOGRAD_AUTOGRAD_H__
#include <memory>
#include <vector>
#include "nn/tensor/tensor.h"
#include "nn/autograd/node.h"

namespace toytorch::autograd {

extern thread_local bool grad_enabled;

class GradModeGuard {
 public:
  explicit GradModeGuard(bool enable) {
    prev_mode_ = grad_enabled;
    grad_enabled = enable;
  }

  ~GradModeGuard() { grad_enabled = prev_mode_; }

 private:
  bool prev_mode_;
};

#define UPDATE_BACKWARD_GRAPH(result, backward_node_name, ...)      \
  do {                                                              \
    if (autograd::grad_enabled &&                                   \
        autograd::is_either_requires_grad({__VA_ARGS__})) {         \
      autograd::update_backward_graph(                              \
          result, std::make_shared<autograd::backward_node_name>(), \
          {__VA_ARGS__});                                           \
    }                                                               \
  } while (false)

#define UPDATE_BACKWARD_GRAPH_1(result, backward_node_name, arg, ...)  \
  do {                                                                 \
    if (autograd::grad_enabled &&                                      \
        autograd::is_either_requires_grad({__VA_ARGS__})) {            \
      autograd::update_backward_graph(                                 \
          result, std::make_shared<autograd::backward_node_name>(arg), \
          {__VA_ARGS__});                                              \
    }                                                                  \
  } while (false)

#define UPDATE_BACKWARD_GRAPH_2(result, backward_node_name, arg1, arg2, ...)  \
  do {                                                                        \
    if (autograd::grad_enabled &&                                             \
        autograd::is_either_requires_grad({__VA_ARGS__})) {                   \
      autograd::update_backward_graph(                                        \
          result, std::make_shared<autograd::backward_node_name>(arg1, arg2), \
          {__VA_ARGS__});                                                     \
    }                                                                         \
  } while (false)

#define BACKWARD_NOT_IMPLEMENTED_YET(opname, ...)             \
  do {                                                        \
    if (autograd::grad_enabled &&                             \
        autograd::is_either_requires_grad({__VA_ARGS__})) {   \
      throw ExceptionOpBackwardNotImplemented(                \
          std::string(#opname) +                              \
          " called on tensors that requires grad hasn't been" \
          " supported yet.");                                 \
    }                                                         \
  } while (false)

#define BACKWARD_NOT_IMPLEMENTED_YET_VEC(opname, args_vec)       \
  do {                                                           \
    if (autograd::grad_enabled &&                                \
        autograd::is_either_requires_grad(args_vec)) {           \
      throw ExceptionOpBackwardNotImplemented(                   \
          std::string(#opname) +                                 \
          " arguments requires grad but its backward logic has " \
          "not been implemented yet");                           \
    }                                                            \
  } while (false)

bool is_either_requires_grad(const std::vector<Tensor>& tensors);

void update_backward_graph(Tensor& result, std::shared_ptr<Node> node,
                           const std::vector<Tensor>& tensors);
void backward(const Tensor& root, Tensor gradient = Tensor(1));

} // namespace toytorch::autograd

#endif // TOYTORCH_NN_AUTOGRAD_AUTOGRAD_H__