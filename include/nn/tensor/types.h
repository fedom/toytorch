#ifndef TOYTORCH_NN_TENSOR_TYPES_H__
#define TOYTORCH_NN_TENSOR_TYPES_H__
#include <vector>

namespace toytorch {

using tt_float = float;
using tt_int = int;

using TensorShape = std::vector<tt_int>;
using TensorIndices = std::vector<tt_int>;

} // namespace toytorch

#endif // TOYTORCH_NN_TENSOR_TYPES_H__