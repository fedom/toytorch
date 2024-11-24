#include "backward_graph_builder.h"
#include <sstream>
#include <unordered_set>
#include "nn/autograd/node.h"

namespace toytorch::debug {

using autograd::Edge;
using autograd::Node;

std::string get_edge_label(const Tensor& tensor) {
  std::ostringstream oss;

  oss << "[label=\"" << tensor.name() << "(";
  std::string delim = "";
  for (int n : tensor.shape()) {
    oss << delim << n;
    delim = ",";
  }
  oss << ")\"";
  if (tensor.requires_grad()) {
    oss << "color=\"red\"";
  }

  oss << "]";
  return oss.str();
}

std::string BackwardGraphBuilder::print_backward_graph(const Tensor& tensor) {
  if (!tensor.requires_grad()) {
    return "";
  }

  const std::string header = "digraph G {";
  const std::string tail = "}";

  std::ostringstream oss;

  oss << header << "\n";
  oss << "start [style=\"invis\"]\n";

  handle_tensor(tensor, "start");

  for (auto& str : nodes_) {
    oss << str << "\n";
  }

  for (auto& edge : edges_) {
    oss << edge << "\n";
  }

  oss << tail << "\n";
  return oss.str();
}

std::string BackwardGraphBuilder::gen_input_node(const Tensor &tensor) {
  return std::string("input_") + std::to_string(reinterpret_cast<uintptr_t>(tensor.raw_data()));
}

void BackwardGraphBuilder::handle_tensor(const Tensor& tensor,
                                         const std::string& src_node_id) {
  std::shared_ptr<Node> node = tensor.grad_info()->grad_fn;
  std::string target_node_id = node->id();

  edges_.push_back(src_node_id + " -> " + target_node_id + " " +
                   get_edge_label(tensor));

  if (nodes_.find(target_node_id) != nodes_.end()) {
    return;
  }

  nodes_.insert(target_node_id);

  for (auto& edge : node->get_edges()) {
    if (edge.get_tensor().requires_grad()) {
      handle_tensor(edge.get_tensor(), target_node_id);
    } else {
      // For input tensor that requires_grad = false, we should display it in the graph
      // though we don't need to recursively handle it.

      std::string input_node = gen_input_node(edge.get_tensor());
      nodes_.insert(input_node + " [style=\"invis\"]");
      edges_.push_back(target_node_id + " -> " + input_node + " " +
                   get_edge_label(edge.get_tensor()));
    }
  }
}

}  // namespace toytorch::debug