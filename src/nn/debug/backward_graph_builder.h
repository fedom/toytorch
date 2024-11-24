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
  void handle_tensor(const Tensor& tensor, const std::string& src_node_id);
  std::string gen_input_node(const Tensor &tensor);


  // key is backward nodes, value is its single output tensor
  // we keep this info to add a tensor node into the edge for 
  // a better visualization. 
  // std::unordered_map<std::string, std::string> nodes_;
  // std::vector<std::string> tensors_;

  std::unordered_set<std::string> nodes_;
  std::vector<std::string> edges_;

};

}  // namespace toytorch::debug

#endif  // SRC_NN_DEBUG_BACKWARD_GRAPH_BUILDER_H__