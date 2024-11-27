#include "nn/operations/convolution.h"
#include "nn/operations/tensor_operations.h"
#include "nn/autograd/autograd.h"
#include "nn/utils/print_utils.h"
#include "nn/exceptions/exceptions.h"
#include <iostream>

namespace toytorch {

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

  if (height_padding != 0 || width_padding != 0) {
    throw ExceptionNotImpl("Padding for conv2d hasn't yet been implemented");
  }

  if (weight_in_channel != input_channel) {
    throw ExceptionInvalidArgument(
        "input channel of input and kernel are not match in conv2d");
  }

  if (height_stride <= 0 || width_stride <= 0 ||
      weight_height > input_height + 2 * height_padding ||
      weight_width > input_width + 2 * width_padding) {
    throw ExceptionInvalidArgument(
        "Height and Width are incompatible of input and kernel in conv2d");
  }

  // Since now we haven't support padding, we don't need to calculate this here which is already calculated
  // in unfold().

  // int result_height = ((input_height + 2 * height_padding - (weight_height - 1) - 1) / height_stride) + 1;
  // int result_width = ((input_width + 2 * width_padding - (weight_width - 1) - 1) / width_stride) + 1;

  // TensorShape result_shape({input_batch, weight_out_channel, result_height, result_width});
  // Tensor result(result_shape);

  // Input shape [N, IN_CH, H, W]
  // Weigtht shape [OUT_CH, IN_CH, KH, KW]
  // Result shape [N, OUT_CH, NEW_H, NEW_W]

  // What we do here is call unfold() to unfold Input's H & W dimension. It results to [N, IN_CH, NEW_H, NEW_W, KH, KW]
  // Compare the new shape with kernel's weight's shape, from convolutoin's calculation rule, we know we should move the
  // IN_CH to the third dim from the end, and IN_CH's position should be left for OUT_CH. In that case, we can simply use
  // existing tensor operation to calculate the whole thing which makes it much easier. Refer to this link for more details
  // on unfold(). https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold

  // new_input's shape is [N, IN_CH, NEW_H, NEW_W, KH, KW]
  Tensor new_input = input.unfold(2, weight_height, height_stride)
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
  Tensor result = new_input * new_weight;
  result = result.sum({4, 5, 6});

  // TODO(Leo): Add UPDATE_BACKWARD_GRAPH(...)
  BACKWARD_NOT_IMPLEMENTED_YET(conv2d, input);

  return result;
}

}  // namespace toytorch