#include "nn/modules/module_impl.h"

namespace toytorch::nn {

std::vector<Tensor> ModuleImpl::parameters(bool recursive) const {
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

std::vector<std::shared_ptr<ModuleImpl>> ModuleImpl::modules(bool recursive) const {
  std::vector<std::shared_ptr<ModuleImpl>> modules;

  for(const auto& item : modules_) {
    modules.push_back(item.second);

    if (recursive) {
      auto nest_modules = item.second->modules(recursive);
      modules.insert(modules.end(), nest_modules.begin(), nest_modules.end());
    }
  }

  return modules;
}

  void ModuleImpl::eval() {
    training_ = false;
    for (auto &item : modules_) {
      item.second->eval();
    }
  }

    void ModuleImpl::train(bool training) {
    training_ = training;
    for (auto &item : modules_) {
      item.second->train(training);
    }
  }

}  // namespace toytorch
