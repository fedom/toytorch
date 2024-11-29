
#include "nn/autograd/backward_node_convolution_op.h"
#include "nn/autograd/autograd.h"
#include "nn/exceptions/exceptions.h"
#include "nn/operations/common_types.h"
#include "nn/operations/convolution.h"
#include "nn/operations/tensor_helper.h"
#include "nn/operations/tensor_operations.h"
#include "nn/tensor/tensor_creator.h"

namespace toytorch::autograd {

namespace {

Tensor full_conv2d(const Tensor& input, const Tensor& weight) {

  int weight_h = weight.dim(-2);
  int weight_w = weight.dim(-1);

  return conv2d(input, weight, {1, 1}, {weight_h - 1, weight_w - 1});
}

Tensor dilate2d(const Tensor& input, int hdilation, int wdilation) {
  if (input.dim() < 2) {
    throw ExceptionInvalidArgument("dilate2d() input dim can't be less than 2");
  }

  // We temporarily set a limit to dilation range
  if (hdilation <= 0 || hdilation > 10 || wdilation <= 0 || wdilation > 10) {
    throw ExceptionInvalidArgument(
        "dilate2d() dilation is out of valid range [1, 10]");
  }

  TensorShape result_shape(input.shape());
  int size = result_shape.size();
  result_shape[size - 2] = (result_shape[size - 2] - 1) * hdilation + 1;
  result_shape[size - 1] = (result_shape[size - 1] - 1) * wdilation + 1;

  Tensor result(result_shape);

  TensorShape shape_first_part(result_shape.begin(), result_shape.end() - 2);
  TensorIndices indices_first_part(shape_first_part.size(), 0);

  int input_height = input.shape()[size - 2];
  int input_width = input.shape()[size - 1];

  do {
    for (int i = 0; i < input_height; i++) {
      for (int j = 0; j < input_width; j++) {
        TensorIndices read_indices_second_part({i, j});
        TensorIndices write_indices_second_part({i * hdilation, j * wdilation});
        TensorIndices read_indices = TensorHelper::merge_indices(
            indices_first_part, read_indices_second_part);
        TensorIndices write_indices = TensorHelper::merge_indices(
            indices_first_part, write_indices_second_part);
        result.at(write_indices) = input.at(read_indices);
      }
    }
  } while (
      TensorHelper::increment_indices(indices_first_part, shape_first_part));

  // This is not intended to support backpropagation for now
  BACKWARD_NOT_IMPLEMENTED_YET("dilate2d", input);

  return result;
}
}  // namespace

// https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
// https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
// ∂L/∂K = conv2d(Input, grad)  -> K : kernel
// ∂L/∂X = full_conv(grad, flip(kernel)) -> X : input
Tensor Conv2dBackward::calculate_lhs_grad(Tensor grad, Tensor input,
                                          Tensor kernel) {

  if (hstride_ > 1 || wstride_ > 1) {
    grad = dilate2d(grad, hstride_, wstride_);
  }

  Tensor fliped_kernel = flip(kernel, {-2, -1});

  // grad shape : [N, OUT, H1, W1]
  // kernel shape: [OUT, IN, H2, W2]
  // input(result) shape: [N, IN, H3, W3]
  // for full_conv2d(grad, flip(kernel)) we need to adjust the shape
  // - kernel [OUT, IN, H2, W2] -> [IN, OUT, H2, W2]
  //
  // Then full_conv2d([N, OUT, H1, W1], [IN, OUT, H2, W2]) -> [N, IN, H3, W3]
  fliped_kernel = fliped_kernel.transpose(0, 1);

  Tensor result = full_conv2d(grad, fliped_kernel);

  int input_grad_h = result.dim(-2);
  int input_grad_w = result.dim(-1);

  if (result.dim(-2) < input.dim(-2) || result.dim(-1) < input.dim(-1)) {
    
    // TODO(Leo): confirm this is the right way to adjust
    result = pad2d(result, 0, input.dim(-2) - result.dim(-2), 0,
                   input.dim(-1) - result.dim(-1));
  }

  return result;
}

Tensor Conv2dBackward::calculate_rhs_grad(Tensor grad, Tensor input,
                                          Tensor kernel) {

  if (hstride_ > 1 || wstride_ > 1) {
    // In this case, we need to dilate the grad. Refer to
    // https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
    grad = dilate2d(grad, hstride_, wstride_);
  }

  // Here we need to adjust the shape in order to use the conv2d().
  //  - input shape: [N, IN_CH, H, W]
  //  - grad shape:  [N, OUT_CH, H1, W1]
  //
  // We need to adjust as below:
  //  - input shape : [N, IN_CH, H, W] -> [IN_CH, N, H, W]
  //  - grad shape :  [N, OUT_CH, H1, W1] -> [OUT_CH, N, H1, W1]
  //
  // After invoke conv2d() on them, we get:
  //  - conv2d(input, grad) -> [IN_CH, OUT_CH, H2, W2]
  //
  // Adjust the result's shape as below:
  // [IN_CH, OUT_CH, H2, W2] -> [OUT_CH, IN_CH, H2, W2]
  //
  // That final result match the original kernel's shape.
  assert(input.dim() == 4);
  assert(grad.dim() == 4);

  Tensor new_input = input.transpose(0, 1);
  Tensor new_grad = grad.transpose(0, 1);
  Tensor result = conv2d(new_input, new_grad);

  // This result's H or W size may greater than the orignal weight's. This is the
  // edge cases that we need to handle. For example (we only use the H and W dimension
  // to illustrate the problem) :
  //
  // In the forward phase:
  //  - input : [5, 5]
  //  - kernel: [2, 3]
  //  - stride: [2, 2]
  // After conv2d(input, kernel), the result shape is:
  //  - output: [2, 2]
  //
  // In the backward phase:
  // - input : [5, 5]
  // - grad_output : [2, 2]
  // - grad_output_dilated : [3, 3]
  // - stride: 1 (Note stride here is not the same with forward pass. The forward pass's stride
  //              is reflected in the dilate() operation as a reverse operation).
  //
  // After conv2d(input, grad_output_dilated):
  // - kernel_grad : [3, 3]
  //
  // Here's the problem, is not the same as the kernel's [2,3] which we expected.
  // Since the semantic of the kernel_grad if the derivative w.r.t the kernels and there is only
  // two rows of the kernel's parameter, this third row parameter doesn't exist actully. But why
  // we got a third row in kernel_grad? The intuition is that, since during the forward pass the
  // input's last row isn't used, but it can be used if there the kernel has 3 rows. If kernel's
  // shape is [3, 3] instead of [2, 3], we still can get the same output's shape as [2, 2]. So in
  // the backward pass, the conv2d(input, grad_output_dilated) actually reflected both cases. And
  // which should be the real case, it leaves to us to decide. Therefore, we can simply discard the
  // extra row.

  int result_h = result.dim(-2);
  int result_w = result.dim(-1);

  int kernel_h = kernel.dim(-2);
  int kernel_w = kernel.dim(-1);

  if (result_h > kernel_h) {
    result = result.slice(-2, 0, kernel_h);
  }
  if (result_w > kernel_w) {
    result = result.slice(-1, 0, kernel_w);
  }

  result = result.transpose(0, 1);

  return result;
}
}  // namespace toytorch::autograd
