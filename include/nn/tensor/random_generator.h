#ifndef TOYTORCH_NN_TENSOR_RANDOM_GENERATOR_H__
#define TOYTORCH_NN_TENSOR_RANDOM_GENERATOR_H__
#include "nn/utils/random_utils.h"

namespace toytorch {

using RandomGeneratorBase = RNGeneratorBase<float>;
using UniformRandomGenerator = UniformRNGenerator<float>;
using NormalRandomGenerator = NormalRNGenerator<float>;


class BernoulliRNGenerator : public RNGeneratorBase<float> {
public:

  BernoulliRNGenerator(float p) : gen_(rd_()), d_(p) {}

  float operator()() override {
    return static_cast<float>(d_(gen_));
  }

private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::bernoulli_distribution d_;
};

}  // namespace toytorch

#endif  // TOYTORCH_NN_TENSOR_RANDOM_GENERATOR_H__