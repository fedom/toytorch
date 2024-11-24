#ifndef TOYTORCH_NN_TENSOR_RANDOM_GENERATOR_H__
#define TOYTORCH_NN_TENSOR_RANDOM_GENERATOR_H__
#include "utils/random_util.h"

namespace toytorch {

using RandomGeneratorBase = RNGeneratorBase<float>;
using UniformRandomGenerator = UniformRNGenerator<float>;
using NormalRandomGenerator = NormalRNGenerator<float>;

}  // namespace toytorch

#endif  // TOYTORCH_NN_TENSOR_RANDOM_GENERATOR_H__