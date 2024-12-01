#include "nn/operations/convolution.h"
#include <iostream>
#include "nn/autograd/autograd.h"
#include "nn/autograd/backward_node_convolution_op.h"
#include "nn/autograd/backward_node_unary_op.h"
#include "nn/exceptions/exceptions.h"
#include "nn/operations/tensor_helper.h"
#include "nn/operations/tensor_operations.h"
#include "nn/utils/print_utils.h"

namespace toytorch {

// https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
Tensor conv2d(const Tensor& input, const Tensor& weight,
              const std::array<int, 2>& stride,
              const std::array<int, 2>& padding) {

  assert(input.dim() == 4);
  assert(weight.dim() == 4);

  int input_batch = input.shape()[0];
  int input_channel = input.shape()[1];
  int input_height = input.shape()[2];
  int input_width = input.shape()[3];

  int weight_out_channel = weight.shape()[0];
  int weight_in_channel = weight.shape()[1];
  int weight_height = weight.shape()[2];
  int weight_width = weight.shape()[3];

  int height_padding = padding[0];
  int width_padding = padding[1];

  int height_stride = stride[0];
  int width_stride = stride[1];

  if (weight_in_channel != input_channel) {
    throw ExceptionInvalidArgument(
        "input channel of input and kernel are not match in conv2d");
  }

  if (height_stride <= 0 || width_stride <= 0) {
    throw ExceptionInvalidArgument(
        "height_stride or width_stride must greater than 0");
  }

  if (weight_height > input_height + 2 * height_padding) {
    throw ExceptionInvalidArgument(
        "weight_height > input_height + 2 * height_padding");
  }
  if (weight_width > input_width + 2 * width_padding) {
    throw ExceptionInvalidArgument(
        "weight_width > input_width + 2 * width_padding");
  }

  Tensor new_input = input;
  if (height_padding != 0 || width_padding != 0) {

    // We put this outside the below lambda to make use of pad2d's auto backward node
    // to handle this padding. So we don't need to implement it again in the conv2d's
    // backward node.
    new_input = pad2d(input, height_padding, height_padding, width_padding, width_padding);
  }

  // new Input shape [N, IN_CH, H, W]
  // Weigtht shape [OUT_CH, IN_CH, KH, KW]
  // Result shape [N, OUT_CH, NEW_H, NEW_W]

  // What we do here is call unfold() to unfold Input's H & W dimension. It results to [N, IN_CH, NEW_H, NEW_W, KH, KW].
  // Comparing the new shape with kernel's weight's shape, from convolutoin's calculation rule, we know we should move the
  // IN_CH to the third dim from the end, and IN_CH's position should be left for OUT_CH. In that case, we can simply use
  // existing tensor operation to calculate the whole thing which makes it much easier. Refer to this link for more details
  // on unfold(). https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold

  // new_input's shape is [N, IN_CH, NEW_H, NEW_W, KH, KW]

  Tensor new_input_copy = new_input;

  Tensor result = [&]() {
    autograd::GradModeGuard grad_guard(false);

    new_input = new_input.unfold(2, weight_height, height_stride)
                    .unfold(3, weight_width, width_stride);

    // shape -> [N, IN_CH, NEW_H, NEW_W, 1, KH, KW]
    new_input = new_input.unsqueeze(4);

    // shape -> [N, 1, NEW_H, NEW_W, IN_CH, KH, KW]
    new_input = new_input.transpose(1, 4);

    // shape -> [OUT_CH, 1, 1, IN_CH, KH, KW]
    Tensor new_weight = weight.unsqueeze(1).unsqueeze(1);

    // shape -> [1, OUT_CH, 1, 1, IN_CH, KH, KW]
    new_weight = new_weight.unsqueeze(0);

    // new_input -> [N,      1, NEW_H, NEW_W, IN_CH, KH, KW]
    // new_weight-> [1, OUT_CH,     1,     1, IN_CH, KH, KW]
    // now we can use element-wise multiply and then sum the last three dims to get the result
    Tensor r = new_input * new_weight;
    r = r.sum({4, 5, 6});

    return r;
  }();

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  // BACKWARD_NOT_IMPLEMENTED_YET("conv2d", input);

  UPDATE_BACKWARD_GRAPH_2(result, Conv2dBackward, height_stride, width_stride,
                          new_input_copy, weight);

  return result;
}

Tensor pad2d(const Tensor& input, int top, int bottom, int left, int right) {

  if (input.dim() < 2) {
    throw ExceptionInvalidArgument(
        "pad2d input should at least has a shape of [H, W]");
  }

  TensorShape result_shape(input.shape());
  result_shape[-2] = result_shape[-2] + top + bottom;
  result_shape[-1] = result_shape[-1] + left + right;

  Tensor result(result_shape);
  TensorIndices result_indices = result.get_indices();
  TensorIndicesWalker result_walker(result_shape, result_indices);
  result_walker.narrow_to_index_range(-2, top, input.shape()[-2]);
  result_walker.narrow_to_index_range(-1, left, input.shape()[-1]);

  TensorIndices read_indices = input.get_indices();
  TensorIndicesWalker read_walker(input.shape(), read_indices);

  do {
    result.at(result_indices) = input.at(read_indices);

  } while (read_walker.step() && result_walker.step());

  UPDATE_BACKWARD_GRAPH_4(result, Pad2dBackward, top, bottom, left, right, input);

  return result;
}

}  // namespace toytorch