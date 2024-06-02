#include "cpu-new-forward.h"

void conv_forward_cpu(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
  /*
  Function parameters:
  output - output
  input - input
  mask - convolution kernel
  Batch - batch_size (number of images in x)
  Map_out - number of output feature maps
  
  Channel - number of input feature maps
  Height - input height dimension
  Width - input width dimension
  K - kernel height and width (K x K)
  */

  const int Height_out = Height - K + 1;
  const int Width_out = Width - K + 1;

  // Example macros:
  // float a = in_4d(0,0,0,0)
  // out_4d(0,0,0,0) = a
  
  #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
  #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
  #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  // Insert CPU convolution kernel code here
  for(int b = 0; b < Batch; b++){
     for(int m = 0; m < Map_out; m++){
        for(int h = 0; h < Height_out; h++){
           for(int w = 0; w < Width_out; w++){
              out_4d(b, m, h, w) = 0;
              for(int c = 0; c < Channel; c++){
                 for(int p = 0; p < K; p++){
                    for(int q = 0; q < K; q++){
                       out_4d(b, m, h, w) += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
                    }
                 }
              }
           }
        }
     }
  }
           
  #undef out_4d
  #undef in_4d
  #undef mask_4d

}
