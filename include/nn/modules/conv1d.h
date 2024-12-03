#ifndef TOYTORCH_NN_MODULES_CONV1D_H__
#define TOYTORCH_NN_MODULES_CONV1D_H__
#include "nn/modules/module.h"
#include "nn/tensor/tensor.h"

namespace toytorch::nn {

class Conv1d : public Module {
 public:
  Conv1d(int in_channels, int out_channels, int kernel_size, int stride = 1,
         const std::array<int, 2>& padding = {0, 0}, bool bias = true);

  Tensor forward(const Tensor& input) const override;

  const Tensor& weights() const { return weights_; }
  const Tensor& bias() const { return bias_; }

  // for test only
  void debug_set_weights(const Tensor& weights) { weights_ = weights; }
  void debug_set_bias(const Tensor& bias) { bias_ = bias; }
  void debug_set_stride(int stride) { stride_ = stride; }
  void debug_set_padding(const std::array<int, 2>& padding) {
    padding_ = padding;
  }

 private:
  Tensor weights_;
  Tensor bias_;
  int stride_;
  std::array<int, 2> padding_;
};

}  // namespace toytorch

#endif  // TOYTORCH_NN_MODULES_CONV1D_H__