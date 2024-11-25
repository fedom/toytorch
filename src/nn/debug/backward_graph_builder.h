#ifndef SRC_NN_DEBUG_BACKWARD_GRAPH_BUILDER_H__
#define SRC_NN_DEBUG_BACKWARD_GRAPH_BUILDER_H__
#include <string>
#include <unordered_set>
#include "nn/tensor/tensor.h"

namespace toytorch::debug {

class BackwardGraphBuilder {
 public:
  std::string print_backward_graph(const Tensor& tensor);

 private:
  void handle_tensor(const Tensor& tensor);
  std::string gen_input_node(const Tensor &tensor);

  std::unordered_set<std::string> backward_op_nodes_;
  std::unordered_map<std::string, std::string> tensor_nodes_;
  std::vector<std::string> edges_;

};

}  // namespace toytorch::debug

#endif  // SRC_NN_DEBUG_BACKWARD_GRAPH_BUILDER_H__