#include "nn/modules/module.h"
#include <gtest/gtest.h>
#include "nn/modules/linear.h"

using namespace toytorch;

class MyNet1 : public nn::Module {
 public:
  MyNet1()
      : fc1_(register_module("fc1", std::make_shared<nn::Linear>(4, 32, "", "fc1"))),
        fc2_(register_module("fc2", std::make_shared<nn::Linear>(32, 1, "", "fc2"))) {}

  Tensor forward(const Tensor& input) const override {
    return input.deep_copy();
  }

 private:
  std::shared_ptr<nn::Linear> fc1_;
  std::shared_ptr<nn::Linear> fc2_;
};

class MyNet2 : public nn::Module {
 public:
  MyNet2() : net1_(register_module("net1", std::make_shared<MyNet1>())), 
             fc_(register_module("fc", std::make_shared<nn::Linear>(16, 4, "", "fc"))) {}

  Tensor forward(const Tensor& input) const override {
    return input.deep_copy();
  }

 private:
  std::shared_ptr<MyNet1> net1_;
  std::shared_ptr<nn::Linear> fc_;
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