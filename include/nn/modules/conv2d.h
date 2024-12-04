#ifndef TOYTORCH_NN_MODULES_CONV2D_H__
#define TOYTORCH_NN_MODULES_CONV2D_H__
#include "nn/modules/module.h"

namespace toytorch::nn {

class Conv2d : public Module {
 public:
  Conv2d(int in_channels, int out_channels,
         const std::array<int, 2>& kernel_size,
         const std::array<int, 2>& stride = {1,1}, const std::array<int, 4>& padding = {0},
         bool bias = true);

  Tensor forward(const Tensor& input) const override;

  const Tensor& weights() const { return weights_; }
  const Tensor& bias() const { return bias_; }

  // for test only
  void debug_set_weights(const Tensor& weights) { weights_.debug_set_value(weights); }
  void debug_set_bias(const Tensor& bias) { bias_.debug_set_value(bias); }
  
  void debug_set_stride(const std::array<int, 2> &stride) {stride_ = stride;}
  void debug_set_padding(const std::array<int, 4> &padding) {padding_ = padding;}

 private:
  Tensor weights_;
  Tensor bias_;
  std::array<int, 4> padding_;
  std::array<int, 2> stride_;
};

}  // namespace toytorch

#endif  // TOYTORCH_NN_MODULES_CONV2D_H__