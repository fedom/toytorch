#ifndef NN_UTILITY_RANDOM_UTILS_H__
#define NN_UTILITY_RANDOM_UTILS_H__
#include <cassert>
#include <concepts>
#include <cstdlib>
#include <ctime>
#include <random>

template <class T, template <typename> class DistType>
struct RNDistrib {
  using type = DistType<T>;
};

template <typename T>
struct UniformRNDistrib;

template <std::integral T>
struct UniformRNDistrib<T> {
  using type = std::uniform_int_distribution<T>;
};

template <std::floating_point T>
struct UniformRNDistrib<T> {
  using type = std::uniform_real_distribution<T>;
};

template <typename T>
using UniformRNDistrib_t = UniformRNDistrib<T>::type;

template <typename T>
using NormalRNDistrib_t = std::normal_distribution<T>;


template <typename T>
class RNGeneratorBase {
  public:
    virtual T operator()() = 0;
};

template <class T>
class UniformRNGenerator : public RNGeneratorBase<T> {
 public:
  UniformRNGenerator(T low, T high) : gen_(rd_()), distrib_(low, high) {}

  T operator()() override { return distrib_(gen_); }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
  UniformRNDistrib_t<T> distrib_;
};

template <std::floating_point T>
class NormalRNGenerator : public RNGeneratorBase<T> {
 public:
  NormalRNGenerator(T mean = 0, T stddev = 1.0)
      : gen_(rd_()), distrib_(mean, stddev) {}

  T operator()() override { return distrib_(gen_); }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
  NormalRNDistrib_t<T> distrib_;
};

#endif  // NN_UTILITY_RANDOM_UTILS_H__
