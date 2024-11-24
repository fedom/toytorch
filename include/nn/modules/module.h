#ifndef SRC_NN_MODULES_MODULE_H__
#define SRC_NN_MODULES_MODULE_H__
#include <unordered_map>
#include "nn/tensor/tensor.h"

namespace toytorch {

class Module {
 public:
  Module() {}
  virtual ~Module() {}

  Module(const Module&) = delete;
  Module(Module&&) = delete;
  Module& operator=(const Module&) = delete;
  Module& operator=(Module&&) = delete;

  virtual Tensor forward(const Tensor& input) const {
    return input.deep_copy();
  }

  // Recursively get all the parameters in this module and submodules
  std::vector<Tensor> parameters(bool recursive = true) const;

  // Register a submodule
  template <typename ModuleType>
  std::shared_ptr<ModuleType> register_module(
      const std::string& name, std::shared_ptr<ModuleType> module) {
    // Register a submodule
    modules_[name] = module;
    return module;
  }

  // Register a parameter
  Tensor& register_parameter(const std::string& name, Tensor tensor);

 protected:
  std::unordered_map<std::string, std::shared_ptr<Module>> modules_;
  std::unordered_map<std::string, Tensor> parameters_;
};

}  // namespace toytorch

#endif  // SRC_NN_MODULES_MODULE_H__