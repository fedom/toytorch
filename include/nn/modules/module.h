#ifndef SRC_NN_MODULES_MODULE_H__
#define SRC_NN_MODULES_MODULE_H__
#include <unordered_map>
#include "nn/tensor/tensor.h"
#include "module_impl.h"

namespace toytorch::nn {

class Module {
 public:
  Module() : impl_(std::make_shared<ModuleImpl>()) {}
  Module(const std::shared_ptr<ModuleImpl> &impl) : impl_(impl) {}

  virtual Tensor forward(const Tensor& input) const {
    return input.deep_copy();
  }

  // Recursively get all the parameters in this module and submodules
  std::vector<Tensor> parameters(bool recursive = true) const {return impl_->parameters();}

  std::vector<Module> modules(bool recursive = true) const;


  void train(bool training = true) {impl_->train();}
  bool is_training() const {return impl_->is_training();}

  void eval() {impl_->eval();}

  // Register a submodule
  template <typename ModuleType>
  ModuleType register_module(
      const std::string& name, ModuleType module) {
        impl_->register_module(name, module.impl());
        return module;
  }

  // Register a parameter
  Tensor& register_parameter(const std::string& name, Tensor tensor) {
    return impl_->register_parameter(name, tensor);
  }

 protected:
  std::shared_ptr<ModuleImpl> impl() const {return impl_;}
  std::shared_ptr<ModuleImpl> impl_;
};

}  // namespace toytorch

#endif  // SRC_NN_MODULES_MODULE_H__