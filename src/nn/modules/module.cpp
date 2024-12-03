#include "nn/modules/module.h"

namespace toytorch {

// Register a parameter
Tensor& Module::register_parameter(const std::string& name, Tensor tensor) {
  parameters_[name] = tensor;
  return parameters_[name];
}

std::vector<Tensor> Module::parameters(bool recursive) const {
  std::vector<Tensor> result;

  for (auto& item : parameters_) {
    result.push_back(item.second);
  }

  if (!recursive) {
    return result;
  }

  // Recursively get all the parameters from submodules
  for (auto& item : modules_) {
    auto tensors = item.second->parameters(recursive);
    result.insert(result.end(), tensors.begin(), tensors.end());
  }

  return result;
}

std::vector<std::shared_ptr<Module>> Module::modules(bool recursive) const {
  std::vector<std::shared_ptr<Module>> modules;

  for(const auto& item : modules_) {
    modules.push_back(item.second);

    if (recursive) {
      auto nest_modules = item.second->modules(recursive);
      modules.insert(modules.end(), nest_modules.begin(), nest_modules.end());
    }
  }

  return modules;
}

}  // namespace toytorch
