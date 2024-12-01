#ifndef TOYTORCH_NN_TENSOR_TYPES_H__
#define TOYTORCH_NN_TENSOR_TYPES_H__
#include "nn/utils/extended_vector.h"

namespace toytorch {

using tt_float = float;
using tt_int = int;

using TensorShape = ExtendedVector<tt_int>;
using TensorIndices = ExtendedVector<tt_int>;

} // namespace toytorch

#endif // TOYTORCH_NN_TENSOR_TYPES_H__