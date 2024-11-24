#ifndef TOYTORCH_NN_AUTOGRAD_NODE_H__
#define TOYTORCH_NN_AUTOGRAD_NODE_H__
#include <vector>
#include "nn/autograd/edge.h"

namespace toytorch::autograd {

#define DEFINE_NODE_NAME_AND_ID(node_name)                    \
  std::string name() const override {                         \
    return #node_name;                                        \
  }                                                           \
  std::string id() const override {                           \
    return std::string(#node_name) + "_" +                    \
           std::to_string(reinterpret_cast<uintptr_t>(this)); \
  }

class Node {
 public:
  Node() {}
  virtual ~Node() {}

  Node(const Node&) = delete;
  Node(Node&&) = delete;
  Node& operator=(const Node&) = delete;
  Node& operator=(Node&&) = delete;

  void increment_in_degree() { in_degree_++; }
  void decrement_in_degree() { in_degree_--; }
  int in_degree() const { return in_degree_; }

  std::vector<Edge>& get_edges() { return edges_; }

  std::vector<Tensor> operator()(std::vector<Tensor>&& input) {
    return apply(std::move(input));
  }

  virtual std::vector<Tensor> apply(std::vector<Tensor>&& input) = 0;
  void add_edge(const Edge& edge) { edges_.push_back(edge); }

  virtual std::string name() const = 0;
  virtual std::string id() const = 0;

 protected:
  std::vector<Edge> edges_;
  int in_degree_;
};

class UnaryNode : public Node {
 public:
  std::vector<Tensor> apply(std::vector<Tensor>&& input) override;
  virtual Tensor calculate_grad(Tensor grad, Tensor input) = 0;
};

class BinaryNode : public Node {
 public:
  std::vector<Tensor> apply(std::vector<Tensor>&& input) override;
  virtual Tensor calculate_lhs_grad(Tensor grad, Tensor lhs, Tensor rhs) = 0;
  virtual Tensor calculate_rhs_grad(Tensor grad, Tensor lhs, Tensor rhs) = 0;
};

}  // namespace toytorch::autograd

#endif  // TOYTORCH_NN_AUTOGRAD_NODE_H__