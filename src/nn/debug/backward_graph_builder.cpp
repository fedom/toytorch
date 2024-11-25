#include "backward_graph_builder.h"
#include <sstream>
#include <unordered_set>
#include "nn/autograd/node.h"

namespace toytorch::debug {

using autograd::Edge;
using autograd::Node;

namespace {

std::string get_tensor_node_style() {
  return "node [shape=circle style=filled fillcolor=aquamarine fontcolor=black fontsize=20]";
}

std::string get_backward_op_node_style() {
  return "node [shape=box style=filled fillcolor=cyan1 fontcolor=black fontsize=20]";
}

std::string get_tensor_shape(const Tensor& tensor) {
  std::ostringstream oss;

  oss << "(";
  std::string delim = "";
  for (int n : tensor.shape()) {
    oss << delim << n;
    delim = ",";
  }
  oss << ")";
  return oss.str();
}

// Copies share the same underlying object have the same identity
std::string get_tensor_node_id(const Tensor& tensor) {
  std::ostringstream oss;
  oss << "tensor_" << tensor.identity();
  return oss.str();
}

std::string get_tensor_node_desc(const Tensor& tensor) {
  std::ostringstream oss;
  oss << "[";
  oss << "label=\"" << get_tensor_shape(tensor) << "\"";
  if (!tensor.requires_grad()) {
    oss << "fillcolor=azure";
  }
  oss << "]";
  return oss.str();
}
}

std::string BackwardGraphBuilder::print_backward_graph(const Tensor& tensor) {
  if (!tensor.requires_grad()) {
    return "";
  }

  tensor_nodes_.insert({get_tensor_node_id(tensor), get_tensor_node_desc(tensor)});
  handle_tensor(tensor);

  std::ostringstream oss;
  oss << "digraph G {\n";

  oss << get_tensor_node_style() << "\n";
  for (auto& [id, desc] : tensor_nodes_) {
    oss << id << " " << desc << "\n";
  }

  oss << get_backward_op_node_style() << "\n";
  for (auto& str : backward_op_nodes_) {
    oss << str << "\n";
  }

  for (auto& edge : edges_) {
    oss << edge << "\n";
  }

  oss << "}\n";
  return oss.str();
}

void BackwardGraphBuilder::handle_tensor(const Tensor& tensor) {
  std::string tensor_node_id = get_tensor_node_id(tensor);

  std::shared_ptr<Node> backward_op_node = tensor.grad_info()->grad_fn;
  std::string backward_op_node_id = backward_op_node->id();

  edges_.push_back(tensor_node_id + " -> " + backward_op_node_id);
  backward_op_nodes_.insert(backward_op_node_id);

  for (auto& edge : backward_op_node->get_edges()) {
    // There will be only one edge comes out from each tensor, while there can be multiples edges
    // come out from one backward_op_node.
    std::string target_tensor_node_id = get_tensor_node_id(edge.get_tensor());
    edges_.push_back(backward_op_node_id + " -> " + target_tensor_node_id);

    if (tensor_nodes_.find(target_tensor_node_id) != tensor_nodes_.end()) {
      continue;
    }
    tensor_nodes_.insert({target_tensor_node_id, get_tensor_node_desc(edge.get_tensor())});

    if (edge.get_tensor().requires_grad()) {
      handle_tensor(edge.get_tensor());
    }
  }
}

}  // namespace toytorch::debug