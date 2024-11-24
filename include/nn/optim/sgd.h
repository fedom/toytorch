#ifndef TOYTORCH_NN_OPTIM_SGD_H__
#define TOYTORCH_NN_OPTIM_SGD_H__
#include "nn/optim/optimizer.h"

namespace toytorch::optim {

class SGD : public Optimizer {
 public:
  SGD(const std::vector<Tensor>& params, float lr = 1e-3, float momentum = 0,
      float dampening = 0, float weight_decay = 0, bool nesterov = false,
      bool maximize = false)
      : Optimizer(params),
        lr_(lr),
        momentum_(momentum),
        dampening_(dampening),
        weight_decay_(weight_decay),
        nesterov_(nesterov),
        maximize_(maximize) {}

  void do_step() override;

 private:

  float lr_;
  float momentum_;
  float dampening_;
  float weight_decay_;
  bool nesterov_;
  bool maximize_;

  bool b_valid_flag_;
  Tensor b_;

};
}  // namespace toytorch::optim

#endif  // TOYTORCH_NN_OPTIM_SGD_H__