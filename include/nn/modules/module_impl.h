#ifndef TOYTORCH_NN_MODULES_MODULE_IMPL_H__
#define TOYTORCH_NN_MODULES_MODULE_IMPL_H__
#include "nn/tensor/tensor.h"
#include <vector>

namespace toytorch::nn {

class ModuleImpl {
 public:
  ModuleImpl() = default;
  virtual ~ModuleImpl() = default;

  ModuleImpl(const ModuleImpl&) = delete;
  ModuleImpl(ModuleImpl&&) = delete;
  ModuleImpl& operator=(const ModuleImpl&) = delete;
  ModuleImpl& operator=(ModuleImpl&&) = delete;

  std::vector<Tensor> parameters(bool recursive = true) const;

  std::vector<std::shared_ptr<ModuleImpl>> modules(bool recursive = true) const;

  // recursive call can't be inlined
  void train(bool training = true);
  void eval();

  inline bool is_training() const {return training_;}

  std::shared_ptr<ModuleImpl> register_module(
      const std::string& name, std::shared_ptr<ModuleImpl> module) {
            // Register a submodule
    modules_[name] = module;
    return module;
      }

  Tensor& register_parameter(const std::string& name, Tensor tensor) {
      parameters_[name] = tensor;
  return parameters_[name];
  }

 protected:
  std::unordered_map<std::string, std::shared_ptr<ModuleImpl>> modules_;
  std::unordered_map<std::string, Tensor> parameters_;
  bool training_;
};

}  // namespace toytorch::nn


#endif // TOYTORCH_NN_MODULES_MODULE_IMPL_H__