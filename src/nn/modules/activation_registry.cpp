#include "nn/modules/activation_registry.h"
#include "exception/exceptions.h"

namespace toytorch {

ActivationRegistry::ActivationRegistry() {}

ActivationRegistry& ActivationRegistry::instance() {
  static ActivationRegistry instance;
  return instance;
}

bool ActivationRegistry::register_activation(const std::string& name,
                                             std::shared_ptr<Activation> act) {

  act_map_[name] = act;
  return true;
}

std::shared_ptr<Activation> ActivationRegistry::get(
    const std::string& name) const {
  if (name.empty()) {
    return nullptr;
  }
  
  auto act = act_map_.find(name);
  if (act == act_map_.end()) {
    throw ExceptionInvalidArgument(
        std::string("unregistered activation name " + name));
  }
  return act->second;
}

}  // namespace toytorch