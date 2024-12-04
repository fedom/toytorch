#include "nn/modules/module.h"
#include <gtest/gtest.h>
#include "nn/modules/linear.h"

using namespace toytorch;

class MyNet1 : public nn::Module {
 public:
  MyNet1()
      : fc1_(nn::Linear(4, 32, "", "fc1")), fc2_(nn::Linear(32, 1, "", "fc2")) {

    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
  }

  Tensor forward(const Tensor& input) const override {
    return input.deep_copy();
  }

 private:
  nn::Linear fc1_;
  nn::Linear fc2_;
};

class MyNet2 : public nn::Module {
 public:
  MyNet2() : net1_(MyNet1()), fc_(nn::Linear(16, 4, "", "fc")) {
    register_module("net1", net1_);
    register_module("fc", fc_);
  }

  Tensor forward(const Tensor& input) const override {
    return input.deep_copy();
  }

 private:
  MyNet1 net1_;
  nn::Linear fc_;
};

TEST(ModuleTest, Parameters) {

  MyNet2 net;

  auto params = net.parameters();

  // for (auto &param : params) {
  //   param.print();
  //   std::cout << "====\n";
  // }

  EXPECT_TRUE(params.size() == 6);
}