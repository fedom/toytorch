#ifndef TOYTORCH_NN_MODULES_ACTIVATION_REGISTRY_H__
#define TOYTORCH_NN_MODULES_ACTIVATION_REGISTRY_H__
#include <unordered_map>
#include "activation.h"

namespace toytorch::nn {

#define REGISTER_ACTIVATION(class_name)                 \
  static bool class_name##_registered = []() {          \
    ActivationRegistry::instance().register_activation( \
        #class_name, std::make_shared<class_name>());   \
    return true;                                        \
  }();

class ActivationRegistry {
 public:
  static ActivationRegistry& instance();

  bool register_activation(const std::string& name,
                           std::shared_ptr<Activation> act);

  std::shared_ptr<Activation> get(const std::string& name) const;

 private:
  ActivationRegistry();

  std::unordered_map<std::string, std::shared_ptr<Activation>> act_map_;
};

}  // namespace toytorch

#endif  // TOYTORCH_NN_MODULES_ACTIVATION_REGISTRY_H__