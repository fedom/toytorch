#include "nn/modules/conv2d.h"
#include <gtest/gtest.h>
#include "nn/autograd/autograd.h"
#include "nn/debug/debug_utils.h"
#include "nn/tensor/tensor_creator.h"
#include "nn/utils/print_utils.h"

using namespace toytorch;

class Conv2dTest : public testing::Test {

 protected:
  constexpr static int input_h = 5;
  constexpr static int input_w = 5;

  constexpr static int batch_size = 2;
  constexpr static int in_channels = 8;
  constexpr static int out_channels = 6;
  constexpr static int kernel_h = 2;
  constexpr static int kernel_w = 3;

  Conv2dTest() : conv_(in_channels, out_channels, {kernel_h, kernel_w}) {

    conv_.debug_set_weights(Tensor(
        {out_channels, in_channels, kernel_h, kernel_w},
        {0.1225,  -0.1128, 0.1288,  0.0126,  -0.1326, 0.1165,  -0.0090, -0.0862,
         0.0242,  0.0481,  0.0758,  0.0378,  0.0852,  -0.0318, 0.0970,  0.1307,
         0.0110,  0.0643,  0.0071,  -0.0285, 0.1433,  0.1063,  -0.1262, -0.0977,
         -0.0563, 0.0526,  0.0842,  0.0435,  0.0589,  0.0675,  -0.0649, -0.1210,
         -0.1074, -0.0126, -0.0080, -0.0683, 0.0133,  -0.0362, -0.0960, 0.1437,
         -0.0535, -0.0656, -0.0483, -0.0778, -0.0088, -0.0181, -0.0297, -0.1092,
         0.0315,  0.0978,  0.0991,  -0.0981, 0.1344,  -0.0795, -0.0084, -0.0819,
         0.0687,  0.0180,  -0.0680, 0.0249,  0.1054,  -0.0506, -0.0511, -0.1323,
         -0.0385, -0.1412, 0.0021,  -0.0188, 0.1230,  0.0660,  0.1373,  0.1226,
         0.0673,  0.0045,  -0.0611, -0.0307, 0.1406,  -0.0265, 0.0715,  0.0553,
         -0.0595, 0.0463,  -0.0643, 0.0406,  -0.0551, 0.0865,  0.1051,  0.0348,
         -0.0815, -0.0998, -0.0756, -0.0450, 0.0959,  0.1034,  0.0899,  0.1338,
         0.0234,  -0.0714, 0.1014,  -0.1121, 0.0605,  0.0004,  0.0406,  -0.0169,
         -0.0002, -0.0170, -0.0152, -0.1014, -0.0635, 0.1270,  -0.0017, 0.1053,
         0.0885,  -0.1022, -0.1175, -0.0196, -0.0114, -0.0823, 0.1153,  -0.1209,
         -0.0718, 0.1387,  0.0184,  0.0057,  0.0639,  -0.0119, 0.0417,  0.0604,
         0.0994,  0.0868,  0.0846,  -0.0663, -0.0935, 0.0129,  0.0143,  0.1434,
         -0.0252, -0.1243, 0.0196,  0.0263,  -0.0087, -0.0002, -0.0157, -0.0773,
         -0.1302, -0.0365, -0.1093, 0.0418,  -0.0762, -0.1094, 0.1126,  -0.0426,
         0.1073,  -0.0002, 0.1426,  -0.0653, 0.0732,  -0.0885, 0.1068,  -0.0784,
         -0.1136, -0.1249, 0.0814,  0.0875,  -0.0498, 0.0969,  -0.0777, -0.0563,
         -0.1047, 0.1269,  -0.0807, 0.0113,  -0.0694, -0.0079, -0.0016, 0.0121,
         -0.0447, 0.0348,  -0.1158, 0.0926,  0.1188,  0.0861,  0.1157,  -0.0794,
         -0.1039, -0.0504, -0.1427, -0.0588, 0.0430,  -0.0961, 0.0492,  -0.0459,
         0.0062,  -0.1042, 0.1122,  0.0956,  0.1154,  -0.1053, 0.1091,  0.0633,
         -0.1001, 0.0545,  -0.0531, -0.0281, -0.1425, -0.0399, 0.1252,  -0.0016,
         0.1140,  0.1269,  0.1150,  0.0423,  0.0439,  -0.1393, 0.0428,  0.0236,
         -0.1085, -0.0964, 0.1351,  -0.0933, 0.0117,  -0.0793, 0.0546,  0.0906,
         -0.1354, 0.0311,  -0.1149, 0.1266,  0.1320,  -0.1188, -0.0053, 0.0429,
         -0.0785, -0.0007, 0.0460,  0.0628,  0.1069,  0.1183,  0.0703,  0.0330,
         -0.0483, 0.0408,  0.1346,  0.0371,  -0.1273, -0.1250, 0.0839,  0.1338,
         0.1158,  -0.0858, 0.0834,  -0.0060, 0.0671,  0.0875,  0.0440,  0.0326,
         0.0736,  0.0961,  0.0213,  -0.0773, 0.1442,  -0.0673, -0.1277, 0.0913,
         0.1052,  0.0119,  -0.0021, -0.1051, 0.0071,  0.0991,  -0.0926, 0.0576,
         -0.1112, -0.0249, 0.1207,  -0.0045, -0.0157, 0.1008,  -0.0073, -0.1239,
         -0.1154, 0.0489,  -0.0790, -0.0802, -0.0507, 0.0205,  0.0253,  0.0259},
        true));

    conv_.debug_set_bias(Tensor(
        {6, 1, 1}, {-0.0309, -0.0957, 0.1231, 0.0454, 0.1217, 0.0560}, true));

    input_ = ones({batch_size, in_channels, input_h, input_w});
  }

  Tensor input_;
  nn::Conv2d conv_;
};

TEST_F(Conv2dTest, forward) {

  Tensor result = conv_.forward(input_);
  // result.print();

  EXPECT_TRUE(result.shape() == TensorShape({batch_size, out_channels, 4, 3}));

  EXPECT_TRUE(result.strict_allclose(
      Tensor({batch_size, out_channels, 4, 3},
             {0.0375,  0.0375,  0.0375,  0.0375,  0.0375,  0.0375,  0.0375,
              0.0375,  0.0375,  0.0375,  0.0375,  0.0375,  0.6431,  0.6431,
              0.6431,  0.6431,  0.6431,  0.6431,  0.6431,  0.6431,  0.6431,
              0.6431,  0.6431,  0.6431,  0.2533,  0.2533,  0.2533,  0.2533,
              0.2533,  0.2533,  0.2533,  0.2533,  0.2533,  0.2533,  0.2533,
              0.2533,  -0.5750, -0.5750, -0.5750, -0.5750, -0.5750, -0.5750,
              -0.5750, -0.5750, -0.5750, -0.5750, -0.5750, -0.5750, 0.8282,
              0.8282,  0.8282,  0.8282,  0.8282,  0.8282,  0.8282,  0.8282,
              0.8282,  0.8282,  0.8282,  0.8282,  0.4890,  0.4890,  0.4890,
              0.4890,  0.4890,  0.4890,  0.4890,  0.4890,  0.4890,  0.4890,
              0.4890,  0.4890,  0.0379,  0.0379,  0.0379,  0.0379,  0.0379,
              0.0379,  0.0379,  0.0379,  0.0379,  0.0379,  0.0379,  0.0379,
              0.6431,  0.6431,  0.6431,  0.6431,  0.6431,  0.6431,  0.6431,
              0.6431,  0.6431,  0.6431,  0.6431,  0.6431,  0.2533,  0.2533,
              0.2533,  0.2533,  0.2533,  0.2533,  0.2533,  0.2533,  0.2533,
              0.2533,  0.2533,  0.2533,  -0.5750, -0.5750, -0.5750, -0.5750,
              -0.5750, -0.5750, -0.5750, -0.5750, -0.5750, -0.5750, -0.5750,
              -0.5750, 0.8282,  0.8282,  0.8282,  0.8282,  0.8282,  0.8282,
              0.8282,  0.8282,  0.8282,  0.8282,  0.8282,  0.8282,  0.4890,
              0.4890,  0.4890,  0.4890,  0.4890,  0.4890,  0.4890,  0.4890,
              0.4890,  0.4890,  0.4890,  0.4890}),
      1e-6, 1e-3));
}

TEST_F(Conv2dTest, forwardWithPadding_2_2) {

  conv_.debug_set_padding({2, 2, 2, 2});
  Tensor result = conv_.forward(input_);
  // result.print();

  EXPECT_TRUE(result.shape() == TensorShape({batch_size, out_channels, 8, 7}));

  EXPECT_TRUE(result.strict_allclose(
      Tensor({batch_size, out_channels, 8, 7},
             {-0.0309, -0.0309, -0.0309, -0.0309, -0.0309, -0.0309, -0.0309,
              -0.0856, -0.2899, 0.1643,  0.1643,  0.1643,  0.2190,  0.4233,
              0.1797,  -0.4663, 0.0375,  0.0375,  0.0375,  -0.1731, 0.4729,
              0.1797,  -0.4663, 0.0375,  0.0375,  0.0375,  -0.1731, 0.4729,
              0.1797,  -0.4663, 0.0375,  0.0375,  0.0375,  -0.1731, 0.4729,
              0.1797,  -0.4663, 0.0375,  0.0375,  0.0375,  -0.1731, 0.4729,
              0.2344,  -0.2073, -0.1577, -0.1577, -0.1577, -0.4230, 0.0187,
              -0.0309, -0.0309, -0.0309, -0.0309, -0.0309, -0.0309, -0.0309,
              -0.0957, -0.0957, -0.0957, -0.0957, -0.0957, -0.0957, -0.0957,
              -0.1208, 0.1291,  0.1365,  0.1365,  0.1365,  0.1616,  -0.0883,
              0.1993,  0.4970,  0.6431,  0.6431,  0.6431,  0.3481,  0.0504,
              0.1993,  0.4970,  0.6431,  0.6431,  0.6431,  0.3481,  0.0504,
              0.1993,  0.4970,  0.6431,  0.6431,  0.6431,  0.3481,  0.0504,
              0.1993,  0.4970,  0.6431,  0.6431,  0.6431,  0.3481,  0.0504,
              0.2244,  0.2722,  0.4109,  0.4109,  0.4109,  0.0908,  0.0430,
              -0.0957, -0.0957, -0.0957, -0.0957, -0.0957, -0.0957, -0.0957,
              0.1231,  0.1231,  0.1231,  0.1231,  0.1231,  0.1231,  0.1231,
              -0.4808, -0.1241, 0.0055,  0.0055,  0.0055,  0.6094,  0.2527,
              -0.2693, 0.3448,  0.2534,  0.2534,  0.2534,  0.6458,  0.0317,
              -0.2693, 0.3448,  0.2534,  0.2534,  0.2534,  0.6458,  0.0317,
              -0.2693, 0.3448,  0.2534,  0.2534,  0.2534,  0.6458,  0.0317,
              -0.2693, 0.3448,  0.2534,  0.2534,  0.2534,  0.6458,  0.0317,
              0.3346,  0.5920,  0.3710,  0.3710,  0.3710,  0.1595,  -0.0979,
              0.1231,  0.1231,  0.1231,  0.1231,  0.1231,  0.1231,  0.1231,
              0.0454,  0.0454,  0.0454,  0.0454,  0.0454,  0.0454,  0.0454,
              -0.3221, -0.6869, -0.7562, -0.7562, -0.7562, -0.3887, -0.0239,
              -0.2338, -0.5124, -0.5749, -0.5749, -0.5749, -0.2957, -0.0171,
              -0.2338, -0.5124, -0.5749, -0.5749, -0.5749, -0.2957, -0.0171,
              -0.2338, -0.5124, -0.5749, -0.5749, -0.5749, -0.2957, -0.0171,
              -0.2338, -0.5124, -0.5749, -0.5749, -0.5749, -0.2957, -0.0171,
              0.1337,  0.2199,  0.2267,  0.2267,  0.2267,  0.1384,  0.0522,
              0.0454,  0.0454,  0.0454,  0.0454,  0.0454,  0.0454,  0.0454,
              0.1217,  0.1217,  0.1217,  0.1217,  0.1217,  0.1217,  0.1217,
              0.2184,  0.3261,  0.4343,  0.4343,  0.4343,  0.3376,  0.2299,
              0.5009,  0.5083,  0.8284,  0.8284,  0.8284,  0.4492,  0.4418,
              0.5009,  0.5083,  0.8284,  0.8284,  0.8284,  0.4492,  0.4418,
              0.5009,  0.5083,  0.8284,  0.8284,  0.8284,  0.4492,  0.4418,
              0.5009,  0.5083,  0.8284,  0.8284,  0.8284,  0.4492,  0.4418,
              0.4042,  0.3039,  0.5158,  0.5158,  0.5158,  0.2333,  0.3336,
              0.1217,  0.1217,  0.1217,  0.1217,  0.1217,  0.1217,  0.1217,
              0.0560,  0.0560,  0.0560,  0.0560,  0.0560,  0.0560,  0.0560,
              0.2818,  0.2215,  -0.0953, -0.0953, -0.0953, -0.3211, -0.2608,
              0.5491,  0.7637,  0.4888,  0.4888,  0.4888,  -0.0043, -0.2189,
              0.5491,  0.7637,  0.4888,  0.4888,  0.4888,  -0.0043, -0.2189,
              0.5491,  0.7637,  0.4888,  0.4888,  0.4888,  -0.0043, -0.2189,
              0.5491,  0.7637,  0.4888,  0.4888,  0.4888,  -0.0043, -0.2189,
              0.3233,  0.5982,  0.6401,  0.6401,  0.6401,  0.3728,  0.0979,
              0.0560,  0.0560,  0.0560,  0.0560,  0.0560,  0.0560,  0.0560,
              -0.0309, -0.0309, -0.0309, -0.0309, -0.0309, -0.0309, -0.0309,
              -0.0856, -0.2899, 0.1643,  0.1643,  0.1643,  0.2190,  0.4233,
              0.1797,  -0.4663, 0.0375,  0.0375,  0.0375,  -0.1731, 0.4729,
              0.1797,  -0.4663, 0.0375,  0.0375,  0.0375,  -0.1731, 0.4729,
              0.1797,  -0.4663, 0.0375,  0.0375,  0.0375,  -0.1731, 0.4729,
              0.1797,  -0.4663, 0.0375,  0.0375,  0.0375,  -0.1731, 0.4729,
              0.2344,  -0.2073, -0.1577, -0.1577, -0.1577, -0.4230, 0.0187,
              -0.0309, -0.0309, -0.0309, -0.0309, -0.0309, -0.0309, -0.0309,
              -0.0957, -0.0957, -0.0957, -0.0957, -0.0957, -0.0957, -0.0957,
              -0.1208, 0.1291,  0.1365,  0.1365,  0.1365,  0.1616,  -0.0883,
              0.1993,  0.4970,  0.6431,  0.6431,  0.6431,  0.3481,  0.0504,
              0.1993,  0.4970,  0.6431,  0.6431,  0.6431,  0.3481,  0.0504,
              0.1993,  0.4970,  0.6431,  0.6431,  0.6431,  0.3481,  0.0504,
              0.1993,  0.4970,  0.6431,  0.6431,  0.6431,  0.3481,  0.0504,
              0.2244,  0.2722,  0.4109,  0.4109,  0.4109,  0.0908,  0.0430,
              -0.0957, -0.0957, -0.0957, -0.0957, -0.0957, -0.0957, -0.0957,
              0.1231,  0.1231,  0.1231,  0.1231,  0.1231,  0.1231,  0.1231,
              -0.4808, -0.1241, 0.0055,  0.0055,  0.0055,  0.6094,  0.2527,
              -0.2693, 0.3448,  0.2534,  0.2534,  0.2534,  0.6458,  0.0317,
              -0.2693, 0.3448,  0.2534,  0.2534,  0.2534,  0.6458,  0.0317,
              -0.2693, 0.3448,  0.2534,  0.2534,  0.2534,  0.6458,  0.0317,
              -0.2693, 0.3448,  0.2534,  0.2534,  0.2534,  0.6458,  0.0317,
              0.3346,  0.5920,  0.3710,  0.3710,  0.3710,  0.1595,  -0.0979,
              0.1231,  0.1231,  0.1231,  0.1231,  0.1231,  0.1231,  0.1231,
              0.0454,  0.0454,  0.0454,  0.0454,  0.0454,  0.0454,  0.0454,
              -0.3221, -0.6869, -0.7562, -0.7562, -0.7562, -0.3887, -0.0239,
              -0.2338, -0.5124, -0.5749, -0.5749, -0.5749, -0.2957, -0.0171,
              -0.2338, -0.5124, -0.5749, -0.5749, -0.5749, -0.2957, -0.0171,
              -0.2338, -0.5124, -0.5749, -0.5749, -0.5749, -0.2957, -0.0171,
              -0.2338, -0.5124, -0.5749, -0.5749, -0.5749, -0.2957, -0.0171,
              0.1337,  0.2199,  0.2267,  0.2267,  0.2267,  0.1384,  0.0522,
              0.0454,  0.0454,  0.0454,  0.0454,  0.0454,  0.0454,  0.0454,
              0.1217,  0.1217,  0.1217,  0.1217,  0.1217,  0.1217,  0.1217,
              0.2184,  0.3261,  0.4343,  0.4343,  0.4343,  0.3376,  0.2299,
              0.5009,  0.5083,  0.8284,  0.8284,  0.8284,  0.4492,  0.4418,
              0.5009,  0.5083,  0.8284,  0.8284,  0.8284,  0.4492,  0.4418,
              0.5009,  0.5083,  0.8284,  0.8284,  0.8284,  0.4492,  0.4418,
              0.5009,  0.5083,  0.8284,  0.8284,  0.8284,  0.4492,  0.4418,
              0.4042,  0.3039,  0.5158,  0.5158,  0.5158,  0.2333,  0.3336,
              0.1217,  0.1217,  0.1217,  0.1217,  0.1217,  0.1217,  0.1217,
              0.0560,  0.0560,  0.0560,  0.0560,  0.0560,  0.0560,  0.0560,
              0.2818,  0.2215,  -0.0953, -0.0953, -0.0953, -0.3211, -0.2608,
              0.5491,  0.7637,  0.4888,  0.4888,  0.4888,  -0.0043, -0.2189,
              0.5491,  0.7637,  0.4888,  0.4888,  0.4888,  -0.0043, -0.2189,
              0.5491,  0.7637,  0.4888,  0.4888,  0.4888,  -0.0043, -0.2189,
              0.5491,  0.7637,  0.4888,  0.4888,  0.4888,  -0.0043, -0.2189,
              0.3233,  0.5982,  0.6401,  0.6401,  0.6401,  0.3728,  0.0979,
              0.0560,  0.0560,  0.0560,  0.0560,  0.0560,  0.0560,  0.0560}),
      1e-6, 1e-3));
}

TEST_F(Conv2dTest, forwardWithStrides_2_2) {

  conv_.debug_set_stride({2, 2});
  Tensor result = conv_.forward(input_);
  EXPECT_TRUE(result.shape() == TensorShape({batch_size, out_channels, 2, 2}));

  EXPECT_TRUE(result.strict_allclose(
      Tensor(
          {batch_size, out_channels, 2, 2},
          {0.0375, 0.0375, 0.0375, 0.0375, 0.6431,  0.6431,  0.6431,  0.6431,
           0.2534, 0.2534, 0.2534, 0.2534, -0.5749, -0.5749, -0.5749, -0.5749,
           0.8284, 0.8284, 0.8284, 0.8284, 0.4888,  0.4888,  0.4888,  0.4888,
           0.0375, 0.0375, 0.0375, 0.0375, 0.6431,  0.6431,  0.6431,  0.6431,
           0.2534, 0.2534, 0.2534, 0.2534, -0.5749, -0.5749, -0.5749, -0.5749,
           0.8284, 0.8284, 0.8284, 0.8284, 0.4888,  0.4888,  0.4888,  0.4888}),
      1e-6, 1e-4));
}

TEST_F(Conv2dTest, forwardWithStrides_2_3) {

  conv_.debug_set_stride({2, 3});
  Tensor result = conv_.forward(input_);
  EXPECT_TRUE(result.shape() == TensorShape({batch_size, out_channels, 2, 1}));

  EXPECT_TRUE(result.strict_allclose(
      Tensor(
          {batch_size, out_channels, 2, 1},
          {0.0375, 0.0375, 0.6431,  0.6431,  0.2534, 0.2534, -0.5749, -0.5749,
           0.8284, 0.8284, 0.4888,  0.4888,  0.0375, 0.0375, 0.6431,  0.6431,
           0.2534, 0.2534, -0.5749, -0.5749, 0.8284, 0.8284, 0.4888,  0.4888}),
      1e-6, 1e-4));
}

TEST_F(Conv2dTest, forwardWithStrides_3_2) {

  conv_.debug_set_stride({3, 2});
  Tensor result = conv_.forward(input_);
  EXPECT_TRUE(result.shape() == TensorShape({batch_size, out_channels, 2, 2}));

  EXPECT_TRUE(result.strict_allclose(
      Tensor(
          {batch_size, out_channels, 2, 2},
          {0.0375, 0.0375, 0.0375, 0.0375, 0.6431,  0.6431,  0.6431,  0.6431,
           0.2534, 0.2534, 0.2534, 0.2534, -0.5749, -0.5749, -0.5749, -0.5749,
           0.8284, 0.8284, 0.8284, 0.8284, 0.4888,  0.4888,  0.4888,  0.4888,
           0.0375, 0.0375, 0.0375, 0.0375, 0.6431,  0.6431,  0.6431,  0.6431,
           0.2534, 0.2534, 0.2534, 0.2534, -0.5749, -0.5749, -0.5749, -0.5749,
           0.8284, 0.8284, 0.8284, 0.8284, 0.4888,  0.4888,  0.4888,  0.4888}),
      1e-6, 1e-4));
}

TEST_F(Conv2dTest, forwardWithStrides_3_3) {

  conv_.debug_set_stride({3, 3});
  Tensor result = conv_.forward(input_);
  EXPECT_TRUE(result.shape() == TensorShape({batch_size, out_channels, 2, 1}));

  EXPECT_TRUE(result.strict_allclose(
      Tensor(
          {batch_size, out_channels, 2, 1},
          {0.0375, 0.0375, 0.6431,  0.6431,  0.2534, 0.2534, -0.5749, -0.5749,
           0.8284, 0.8284, 0.4888,  0.4888,  0.0375, 0.0375, 0.6431,  0.6431,
           0.2534, 0.2534, -0.5749, -0.5749, 0.8284, 0.8284, 0.4888,  0.4888}),
      1e-6, 1e-4));
}

TEST_F(Conv2dTest, forwardWithPadding_2_2_Strides_2_3) {

  conv_.debug_set_padding({2, 2, 2, 2});
  conv_.debug_set_stride({2, 3});
  Tensor result = conv_.forward(input_);
  EXPECT_TRUE(result.shape() == TensorShape({batch_size, out_channels, 4, 3}));

  EXPECT_TRUE(result.strict_allclose(
      Tensor({batch_size, out_channels, 4, 3},
             {-0.0309, -0.0309, -0.0309, 0.1797,  0.0375,  0.4729,  0.1797,
              0.0375,  0.4729,  0.2344,  -0.1577, 0.0187,  -0.0957, -0.0957,
              -0.0957, 0.1993,  0.6431,  0.0504,  0.1993,  0.6431,  0.0504,
              0.2244,  0.4109,  0.0430,  0.1231,  0.1231,  0.1231,  -0.2693,
              0.2534,  0.0317,  -0.2693, 0.2534,  0.0317,  0.3346,  0.3710,
              -0.0979, 0.0454,  0.0454,  0.0454,  -0.2338, -0.5749, -0.0171,
              -0.2338, -0.5749, -0.0171, 0.1337,  0.2267,  0.0522,  0.1217,
              0.1217,  0.1217,  0.5009,  0.8284,  0.4418,  0.5009,  0.8284,
              0.4418,  0.4042,  0.5158,  0.3336,  0.0560,  0.0560,  0.0560,
              0.5491,  0.4888,  -0.2189, 0.5491,  0.4888,  -0.2189, 0.3233,
              0.6401,  0.0979,  -0.0309, -0.0309, -0.0309, 0.1797,  0.0375,
              0.4729,  0.1797,  0.0375,  0.4729,  0.2344,  -0.1577, 0.0187,
              -0.0957, -0.0957, -0.0957, 0.1993,  0.6431,  0.0504,  0.1993,
              0.6431,  0.0504,  0.2244,  0.4109,  0.0430,  0.1231,  0.1231,
              0.1231,  -0.2693, 0.2534,  0.0317,  -0.2693, 0.2534,  0.0317,
              0.3346,  0.3710,  -0.0979, 0.0454,  0.0454,  0.0454,  -0.2338,
              -0.5749, -0.0171, -0.2338, -0.5749, -0.0171, 0.1337,  0.2267,
              0.0522,  0.1217,  0.1217,  0.1217,  0.5009,  0.8284,  0.4418,
              0.5009,  0.8284,  0.4418,  0.4042,  0.5158,  0.3336,  0.0560,
              0.0560,  0.0560,  0.5491,  0.4888,  -0.2189, 0.5491,  0.4888,
              -0.2189, 0.3233,  0.6401,  0.0979}),
      1e-6, 1e-4));
}