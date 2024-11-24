#ifndef TOYTORCH_NN_OPTIM_OPTIMIZER_H__
#define TOYTORCH_NN_OPTIM_OPTIMIZER_H__
#include "nn/tensor/tensor.h"
#include "nn/autograd/autograd.h"

namespace toytorch {
  namespace optim {
    class Optimizer {
      public:
        Optimizer(const std::vector<Tensor> &params) : params_(params) {}

        void zero_grad() {
          for (auto &param : params_) {
            param.grad_info()->grad = std::make_shared<Tensor>(0);
          }
        }

        void step() {
          autograd::GradModeGuard grad_guard(false);
          do_step();
        }

        virtual void do_step() = 0;

      protected:
        std::vector<Tensor> params_;
    };
  }

} // namespace toytorch

#endif // TOYTORCH_NN_OPTIM_OPTIMIZER_H__