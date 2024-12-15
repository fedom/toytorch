#ifndef TOYTORCH_NN_OPERATIONS_ACTIVATIONS_H__
#define TOYTORCH_NN_OPERATIONS_ACTIVATIONS_H__
#include "nn/tensor/tensor.h"

namespace toytorch {

Tensor sigmoid(const Tensor& tensor);
Tensor relu(const Tensor& tensor);
Tensor softmax(const Tensor& input, int dim);
Tensor log_softmax(const Tensor& input, int dim);

} // namespace toytorch

#endif // TOYTORCH_NN_OPERATIONS_ACTIVATIONS_H__