#ifndef TOYTORCH_NN_OPTIM_SGD_H__
#define TOYTORCH_NN_OPTIM_SGD_H__
#include "nn/optim/optimizer.h"

namespace toytorch::optim {

class SGDOptions {
 public:
  SGDOptions() = default;
  SGDOptions(float lr) : lr_(lr) {}

  inline SGDOptions& lr(const float lr_v) {
    lr_ = lr_v;
    return *this;
  }
  inline float lr() const { return lr_; }

  inline SGDOptions& momentum(const float momentum_v) {
    momentum_ = momentum_v;
    return *this;
  }

  inline float momentum() const { return momentum_; }

  inline SGDOptions& dampening(const float dampening_v) {
    dampening_ = dampening_v;
    return *this;
  }
  inline float dampening() const { return dampening_; }

  inline SGDOptions& weight_decay(const float weight_decay_v) {
    weight_decay_ = weight_decay_v;
    return *this;
  }
  inline float weight_decay() const { return weight_decay_; }

  inline SGDOptions& nesterov(const bool nesterov_v) {
    nesterov_ = nesterov_v;
    return *this;
  }
  inline bool nesterov() const { return nesterov_; }

  inline SGDOptions& maximize(const bool maximize_v) {
    maximize_ = maximize_v;
    return *this;
  }
  inline bool maximize() const { return maximize_; }

  float lr_;
  float momentum_;
  float dampening_;
  float weight_decay_;
  bool nesterov_;
  bool maximize_;
};

class SGD : public Optimizer {
 public:
  SGD(const std::vector<Tensor>& params, const SGDOptions& options)
      : Optimizer(params),
        lr_(options.lr()),
        momentum_(options.momentum()),
        dampening_(options.dampening()),
        weight_decay_(options.weight_decay()),
        nesterov_(options.nesterov()),
        maximize_(options.maximize()) {}

  void do_step() override;

 private:
  float lr_;
  float momentum_;
  float dampening_;
  float weight_decay_;
  bool nesterov_;
  bool maximize_;

  // keep internal state between steps
  bool b_valid_flag_;
  Tensor b_;
};
}  // namespace toytorch::optim

#endif  // TOYTORCH_NN_OPTIM_SGD_H__