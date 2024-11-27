#ifndef TOYTORCH_NN_UTILS_PRINT_UTILS_H__
#define TOYTORCH_NN_UTILS_PRINT_UTILS_H__

#include "nn/tensor/types.h"
#include <vector>
#include <iostream>

inline std::ostream& operator<<(std::ostream &os, const toytorch::TensorShape &vec) {
  os << "(";
  for (auto &v : vec) {
    os << v << ",";
  }
  os << ")" << std::endl;
  return os;
}

#endif // TOYTORCH_NN_UTILS_PRINT_UTILS_H__