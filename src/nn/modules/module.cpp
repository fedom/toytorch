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

}  // namespace toytorch
