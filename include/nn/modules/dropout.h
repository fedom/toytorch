#ifndef TOYTORCH_NN_MODULES_DROPOUT_H__
#define TOYTORCH_NN_MODULES_DROPOUT_H__
#include "nn/modules/module.h"

namespace toytorch::nn {

class Dropout : public Module {
public:
  Dropout(float p) : p_(p) {}

  Tensor forward(const Tensor& input) const override;

private:
  float p_;
};

class Dropout2d : public Module {
public:
  Dropout2d(float p) : p_(p) {}

  Tensor forward(const Tensor& input) const override;

private:
  float p_;

};

} // namespace toytorch

#endif // TOYTORCH_NN_MODULES_DROPOUT_H__