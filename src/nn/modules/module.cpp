#include "nn/modules/module.h"

namespace toytorch::nn {

std::vector<Module> Module::modules(bool recursive) const {
  std::vector<std::shared_ptr<ModuleImpl>> modules = impl_->modules(recursive);

  std::vector<Module> output(modules.size());
  for (int i = 0; i < modules.size(); i++) {
    output[i] = Module(modules[i]);
  }
  return output;
}

}  // namespace toytorch
