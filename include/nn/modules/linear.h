#ifndef SRC_NN_MODULES_LINEAR_H__
#define SRC_NN_MODULES_LINEAR_H__
#include <memory>
#include <string>
#include <vector>
#include "nn/modules/activation.h"
#include "nn/modules/activation_registry.h"
#include "nn/modules/module.h"
#include "nn/tensor/tensor.h"

namespace toytorch {

class Linear : public Module {
 public:
  Linear(int num_input, int num_output,
              const std::string &act_name = "",
              const std::string &name = "");

  Tensor forward(const Tensor& input) const override;

  const Tensor &weights() const { return weights_; }
  const Tensor &bias() const { return bias_; }
  const std::string& get_name() const { return name_; }

  // for test only
  void debug_set_weights(const Tensor &weights) { weights_ = weights; }
  void debug_set_bias(const Tensor &bias) { bias_ = bias; }

 private:
  Tensor weights_;
  Tensor bias_;

  std::shared_ptr<Activation> activation_;
  std::string name_;
};

}  // namespace toytorch

#endif  // SRC_NN_MODULES_LINEAR_H__