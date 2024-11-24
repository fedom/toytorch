#include "nn/debug/debug_utils.h"
#include "backward_graph_builder.h"

namespace toytorch::debug {

std::string print_backward_graph(Tensor tensor) {
  BackwardGraphBuilder builder;

  return builder.print_backward_graph(tensor);
}

} // namespace toytorch::debug

