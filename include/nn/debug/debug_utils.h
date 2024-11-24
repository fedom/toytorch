#ifndef TOYTORCH_NN_DEBUG_DEBUG_UTILS_H__
#define TOYTORCH_NN_DEBUG_DEBUG_UTILS_H__

#include "nn/tensor/tensor.h"

namespace toytorch::debug {

std::string print_backward_graph(Tensor tensor);

} // namespace toytorch::debug

#endif // TOYTORCH_NN_DEBUG_DEBUG_UTILS_H__