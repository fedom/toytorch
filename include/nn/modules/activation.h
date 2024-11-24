#ifndef SRC_NN_MODULES_ACTIVATION_H__
#define SRC_NN_MODULES_ACTIVATION_H__
#include "module.h"

namespace toytorch {

class Activation : public Module {

};

class Sigmoid : public Activation {
 public:
  Tensor forward(const Tensor& input) const override;
};

class Relu : public Activation {
 public:
  Tensor forward(const Tensor& input) const override;
};

}  // namespace toytorch

#endif  // SRC_NN_MODULES_ACTIVATION_H__