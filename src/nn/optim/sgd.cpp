#include "nn/optim/sgd.h"
#include "nn/tensor/tensor_operations.h"

namespace toytorch::optim {

// https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
void SGD::do_step() {
  for (auto &param : params_) {
    Tensor grad = *param.grad();
    if (weight_decay_) {
      grad = grad + weight_decay_ * param;
    }
    if (momentum_) {
      if (b_valid_flag_) {
        b_ = momentum_ * b_ + (1 - dampening_) * grad;
      } else {
        b_ = grad;
        b_valid_flag_ = true;
      }

      if (nesterov_) {
        grad = grad + momentum_ * b_;
      } else {
        grad = b_;
      }
    }

    if (maximize_) {
      param.add_(lr_ * grad);
    } else {
      param.sub_(lr_ * grad);
    }
  }
  
}

}  // namespace toytorch::optim