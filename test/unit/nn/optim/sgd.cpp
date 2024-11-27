#include <gtest/gtest.h>
#include "nn/operations/tensor_helper.h"
#include "nn/operations/tensor_operations.h"
#include "nn/tensor/tensor_creator.h"
#include "nn/exceptions/exceptions.h"
#include "nn/optim/sgd.h"

using namespace toytorch;

/*
import torch


a = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, 3, requires_grad=True)
input = torch.randn(2, 3)
print(f"{a=}")
print(f"{b=}")
print(f"{input=}")

c = a * input + b
d = c.sum()

sgd = torch.optim.SGD([a, b], 0.001)
d.backward()

print(f"{a.grad=}")
print(f"{b.grad=}")

sgd.step()
print(f"{a=}")
print(f"{b=}")

sgd.zero_grad()

print("==============round 2================")

c = a * input + b
d = c.sum()

sgd = torch.optim.SGD([a, b], 0.001)
d.backward()

print(f"{a.grad=}")
print(f"{b.grad=}")

sgd.step()
print(f"{a=}")
print(f"{b=}")

# ===========output============
a=tensor([[-3.9687e-01,  6.3874e-01, -3.6563e-01],
        [ 1.6797e+00, -1.6143e+00, -9.5886e-04]], requires_grad=True)
b=tensor([[-0.9223, -1.8733,  0.6321],
        [-1.4158,  1.2890,  0.1078]], requires_grad=True)
input=tensor([[-0.3619, -0.1521,  0.3433],
        [-1.1516,  0.1207,  1.5593]])
a.grad=tensor([[-0.3619, -0.1521,  0.3433],
        [-1.1516,  0.1207,  1.5593]])
b.grad=tensor([[1., 1., 1.],
        [1., 1., 1.]])
a=tensor([[-0.3965,  0.6389, -0.3660],
        [ 1.6809, -1.6145, -0.0025]], requires_grad=True)
b=tensor([[-0.9233, -1.8743,  0.6311],
        [-1.4168,  1.2880,  0.1068]], requires_grad=True)
==============round 2================
a.grad=tensor([[-0.3619, -0.1521,  0.3433],
        [-1.1516,  0.1207,  1.5593]])
b.grad=tensor([[1., 1., 1.],
        [1., 1., 1.]])
a=tensor([[-0.3961,  0.6390, -0.3663],
        [ 1.6820, -1.6146, -0.0041]], requires_grad=True)
b=tensor([[-0.9243, -1.8753,  0.6301],
        [-1.4178,  1.2870,  0.1058]], requires_grad=True)
*/

TEST(OptimTest, sgd) {

  Tensor p1 = Tensor({2, 3}, {-3.9687e-01,  6.3874e-01, -3.6563e-01, 1.6797e+00, -1.6143e+00, -9.5886e-04}, true);
  Tensor p2 = Tensor({2, 3}, {-0.9223, -1.8733,  0.6321, -1.4158,  1.2890,  0.1078}, true);
  Tensor input = Tensor({2, 3}, {-0.3619, -0.1521,  0.3433, -1.1516,  0.1207,  1.5593});

  optim::SGD sgd({p1, p2}, 0.001);

  Tensor c = p1 * input + p2;
  Tensor d = c.sum();
  d.backward();

  EXPECT_TRUE(p1.grad()->strict_allclose(Tensor({2, 3}, {-0.3619, -0.1521,  0.3433, -1.1516,  0.1207,  1.5593}), 1e-6, 1e-4));
  EXPECT_TRUE(p2.grad()->strict_allclose(Tensor({2, 3}, {1., 1., 1., 1., 1., 1.}), 1e-6, 1e-4));

  sgd.step();

  EXPECT_TRUE(p1.strict_allclose(Tensor({2, 3}, {-0.3965,  0.6389, -0.3660, 1.6809, -1.6145, -0.0025}), 1e-6, 1e-4));
  EXPECT_TRUE(p2.strict_allclose(Tensor({2, 3}, {-0.9233, -1.8743,  0.6311, -1.4168,  1.2880,  0.1068}), 1e-6, 1e-4));

  sgd.zero_grad();

  EXPECT_TRUE(*p1.grad() == Tensor(0));
  EXPECT_TRUE(*p2.grad() == Tensor(0));

  c = p1 * input + p2;
  d = c.sum();
  d.backward();

  EXPECT_TRUE(p1.grad()->strict_allclose(Tensor({2, 3}, {-0.3619, -0.1521,  0.3433, -1.1516,  0.1207,  1.5593}), 1e-6, 1e-4));
  EXPECT_TRUE(p2.grad()->strict_allclose(Tensor({2, 3}, {1., 1., 1., 1., 1., 1.}), 1e-6, 1e-4));

  sgd.step();

  EXPECT_TRUE(p1.strict_allclose(Tensor({2, 3}, {-0.3961,  0.6390, -0.3663, 1.6820, -1.6146, -0.0041}), 1e-6, 1e-4));
  EXPECT_TRUE(p2.strict_allclose(Tensor({2, 3}, {-0.9243, -1.8753,  0.6301, -1.4178,  1.2870,  0.1058}), 1e-6, 1e-4));
}