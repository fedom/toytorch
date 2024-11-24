
#include "nn/autograd/autograd.h"
#include "nn/autograd/backward_node_leaf_op.h"
#include "nn/autograd/edge.h"
#include "nn/tensor/tensor_operations.h"
#include "exception/exceptions.h"
#include <queue>

namespace toytorch::autograd {

thread_local bool grad_enabled = true;

bool is_either_requires_grad(const std::vector<Tensor>& tensors) {
  for (auto& tensor : tensors) {
    if (tensor.requires_grad()) {
      return true;
    }
  }
  return false;
}

void update_backward_graph(Tensor& result, std::shared_ptr<Node> node,
                           const std::vector<Tensor>& tensors) {

  // caller need do this check
  assert(is_either_requires_grad(tensors));

  // If either one of the two operands requires grad, the result need grad
  result.init_grad_info();

  for (auto& tensor : tensors) {
    // We need to add the input tensor as an edge even if the tensor doesn't require grad.
    // This is because when calculating grad in grad_fn, we extract the input tensors
    // from its edges. If even if the tensor itself doesn't need to calculate grad, it may
    // be needed for calculating its siblings's grad.
    node->add_edge(Edge(tensor));

    if (tensor.requires_grad()) {
      assert(tensor.grad_info());

      // Since tensor's grad_fn is set when it is created as a result of an operation,
      // leaf tensor (which isn't a result of operations) has no chance to be set. So
      // we check and set it when we connect the input tensors as edges.
      if (!tensor.grad_info()->grad_fn) {
        tensor.grad_info()->grad_fn =
            std::make_shared<autograd::LeafNodeBackward>();
      }
      tensor.grad_info()->grad_fn->increment_in_degree();
    }
  }

  result.grad_info()->grad_fn = node;
}

struct GraphTask {
  std::shared_ptr<Node> node;
  std::shared_ptr<Tensor> input;
};

void backward(const Tensor& root, Tensor gradient) {

  GradModeGuard grad_guard(false);

  if (!root.is_scalar() && root.shape() != gradient.shape()) {
    throw ExceptionInvalidArgument(
        "backward() is only supported on scalar tensor now");
  }

  std::queue<GraphTask> task_queue;

  assert(root.grad_info() && root.grad_info()->grad_fn);

  task_queue.push(
      {root.grad_info()->grad_fn, std::make_shared<Tensor>(gradient)});

  while (!task_queue.empty()) {
    GraphTask task = task_queue.front();
    task_queue.pop();

    std::vector<Tensor> result = task.node->apply({*task.input});

    auto& edges = task.node->get_edges();
    for (int i = 0; i < edges.size(); i++) {
      if (!edges[i].get_tensor().requires_grad())
        continue;

      std::shared_ptr<GradInfo> grad_info = edges[i].get_tensor().grad_info();
      assert(grad_info);

      *grad_info->grad = add(*grad_info->grad, result[i]);

      edges[i].get_node()->decrement_in_degree();
      if (!edges[i].get_node()->in_degree()) {
        task_queue.push({edges[i].get_node(), edges[i].get_tensor().grad()});
      }
    }
  }
}

}  // namespace toytorch::autograd