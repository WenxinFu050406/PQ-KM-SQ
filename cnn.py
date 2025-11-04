"""
https://discuss.pytorch.org/t/manual-convolution-does-not-match-results-of-pytorch-conv2d-floating-point/141192

def conv2d_matmul_fp(sample_input, weight, padding, stride, dilation):
    N,C,X,Y = sample_input.size()
    K,_,R,S = weight.size()

    out_size = (math.floor((X+padding[0]*2-dilation[0]*(R-1)-1)/stride[0]) + 1, math.floor((Y+padding[1]*2-dilation[1]*(S-1)-1)/stride[1]) + 1)

    simple_in_unfold = torch.nn.functional.unfold(sample_input, kernel_size=(R,S), dilation=dilation, padding=padding, stride=stride)
    res = torch.matmul(weight.view(weight.size()[0], -1), simple_in_unfold[0])
    return res.reshape(N, K, out_size[0], out_size[1])

https://cs231n.github.io/convolutional-networks/




"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
# import pretty_errors
from utils import Timer



class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
        super(MyConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

        self.weights = None  # nn.Parameter(torch.randn((self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])))
        self.bias = None  # nn.Parameter(torch.randn(self.out_channels))
        
    
    def forward(self, x):
        y = myConv2d(
            x, self.weights, self.out_channels, 
            stride=self.stride, padding=self.padding, 
            dilation=self.dilation, bias=self.bias
        )
        return y





def myConv2d(x, kernel, channel_out, stride=1, padding=1, dilation=1, bias=None):
    """
    tensor shape: BCHW
    kernel shape: channel_out, channel_in, kernel_height, kernel_width

    input tensor x is padded with 0s on all 4 sides
    the convolution window moves with the kernel size from the padding
    - loop indexes are centered for the kernel filter
    """
    # check shapes
    assert len(x.shape) == 4, f'expected 4-D tensor, got {len(x.shape)}-D tensor'
    assert len(kernel.shape) == 4, f'expected 4-D tensor shape, got {len(kernel.shape)}-D tensor'
    assert kernel.shape[2] == kernel.shape[3], f'expected square kernel shape, got {len(kernel.shape)}-D tensor'

    # assert stride == 1, f'stride > 1 not implemented yet'
    # assert padding == 1, f'padding > 1 not implemented yet'

    kernel = kernel.cpu()  # .cuda(0)
    bias = bias.cpu()  # .cuda(0)
    if isinstance(padding, tuple):
        padding = padding[0]

    assert dilation == 1, f'dilation > 1 not implemented yet'
    # assert bias is False, 'bias is not implemented yet'


    # kernel side
    ksh, ksw = kernel.shape[2], kernel.shape[3]
    
    B, channel_in, H, W = x.size()
    p2 = padding * 2

    # x_padded = torch.zeros((B, channel_in, H + 2 * padding, W + 2 * padding)).to(x.device)
    # x_padded[:, :, padding: H + padding, padding: W + padding] = x
    # print('x_padded.shape =', x_padded.shape)

    # out = torch.zeros((
    #     B, channel_out,
    #     (H - ksh + p2)//stride + 1,
    #     (W - ksw + p2)//stride + 1
    # )).to(x.device)

    out2 = torch.zeros((
        B, channel_out,
        (H - ksh + p2)//stride + 1,
        (W - ksw + p2)//stride + 1
    )).to(x.device)

    # Hp, Wp = out.size()[-2:]

    for b in range(B):
        for c_out in range(channel_out):
            # out[b, c_out, :, :] = bias[c_out] if bias is not None else 0.  # accumulation of the scalar product
            # print(c_out)
            for c_in in range(channel_in):
                f = kernel[c_out, c_in]  # one single kernel filter
                # print(c_out, c_in)
                # convolution, 2d window sliding

                # for h in range(Hp):
                #     hst = h * stride
                #     for w in range(Wp):
                #         wst = w * stride
                #         window_2d = x_padded[b, c_in, hst: hst+ksh, wst: wst+ksw]

                #         scalar_prod = torch.sum(window_2d * f)
                #         out[b, c_out, h, w] += scalar_prod


                x_rs = x[b, c_in].unsqueeze(0).unsqueeze(0)
                f_rs = f.reshape(1, 1, 3, 3)

                out2[b, c_out] += nn.functional.conv2d(x_rs, f_rs, bias=None, stride=stride, padding=padding).squeeze()
                # assert torch.max(torch.abs(out - out2)) < 1e-6


    if bias is not None:
        for c_out in range(channel_out):
            # out[:, c_out, :, :] += bias[c_out]
            out2[:, c_out, :, :] += bias[c_out]

            # print(out[:, c_out, :, :])
            # print(bias[c_out])
            
    return out2








@torch.no_grad()
def main():
    B = 10
    k = 3
    p = 0
    stride = 1
    
    channel_in = 3
    channel_out = 64
    x = torch.ones(B, channel_in, 32, 32).cuda(0)

    device = torch.device('cpu')
    x = x.to(device)


    # y = torch.randn(1, 32, 28, 28)
    # z = torch.randn(1, 32, 28, 28)
    print(x.shape)
    # print(y.shape)
    # print(z.shape)
    conv = nn.Conv2d(channel_in, channel_out,
        kernel_size=k, stride=stride, padding=p, bias=True).to(device)
    conv.eval()
    # w, b = list(conv.parameters())
    # print(w, w.shape)

    timer = Timer()
    timer.start()
    x2 = conv(x)
    timer.stop()
    timer.print_duration(msg='nn.Conv2d 1')

    timer = Timer()
    timer.start()
    x2 = conv(x)
    timer.stop()
    timer.print_duration(msg='nn.Conv2d 2')

    timer = Timer()
    timer.start()
    x2_1 = myConv2d(x, conv.weight, channel_out, 
        padding=p, stride=stride, bias=conv.bias)
    timer.stop()
    timer.print_duration(msg='myConv2d')


    timer = Timer()
    timer.start()
    x2_1 = myConv2d(x, conv.weight, channel_out, 
        padding=p, stride=stride, bias=conv.bias)
    timer.stop()
    timer.print_duration(msg='myConv2d second run')


    device = torch.device('cpu')

    x = x.to(device)

    total_time_conv = 0
    total_time_convp = 0
    num_samples_conv = 0
    num_samples_convp = 0
    while True:



        timer0 = Timer()
        timer0.start()
        x2_1 = myConv2d(x, conv.weight.cpu(), channel_out, 
            padding=p, stride=stride, bias=conv.bias.cpu())
        timer0.stop()
        # timer.print_duration(msg='myConv2d second run')
        total_time_conv += timer0.duration
        num_samples_conv += 1
        average_conv = total_time_conv / num_samples_conv * 1000





if __name__ == '__main__':
    main()
