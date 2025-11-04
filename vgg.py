'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from cnn import MyConv2d


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

def allclose(x, y):
    return torch.allclose(x, y)


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}



class VGG19(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, mode='conv2d'):
        """
        conv2d: pytorch original conv2d
        myconv2d: conv2d implemented by myself
        myconv2d_p: conv2d based on permutations
        """
        super(VGG19, self).__init__()
        # VGG(make_layers(cfg['E']))

        assert mode in ['conv2d', 'myconv2d', 'myconv2d_p']
        self.mode = mode
        self.cfg19 = cfg['E']

        self.layers = []
        self.layers_myconv2d = []

        in_channels = 3
        for v in self.cfg19:
            if v == 'M':
                self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:

                if self.mode == 'conv2d':
                    conv_obj = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                elif self.mode == 'myconv2d':
                    conv_obj = MyConv2d(in_channels, v, kernel_size=3, padding=1)
                

                self.layers.append(conv_obj)
                self.layers.append(nn.ReLU(inplace=True))

                in_channels = v



        self.features = nn.Sequential(*self.layers)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def set_myconv2d_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                w, b = layer.weight, layer.bias

                l = MyConv2d(layer.in_channels, layer.out_channels, kernel_size=layer.kernel_size, padding=layer.padding)
                l.weights = w
                l.bias = b
                self.layers_myconv2d.append(l)
            else:
                self.layers_myconv2d.append(layer)

    def forward_myconv2d(self, x):
        for layer in self.layers_myconv2d:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def compare_layers_same_inputs(self, x):
        """
        compare layers, each layer with same input, not like a normal forward pass
        """
        x1, x2 = x, x
        y = x
        with torch.no_grad():
            for l1, l2 in zip(self.layers, self.layers_myconv2d):
                x1 = l1(y)
                x2 = l2(y)
                y = x1
                # print(allclose(x1, x2))
                inf_norm = torch.max(torch.abs(x1 - x2))
                print(l1)
                print(l2)
                print(inf_norm, inf_norm < 1e-3)
                print()

    def compare_layers_forward(self, x):
        """
        compare layers in forward mode, errors may accumulate layer by layer
        """
        x1, x2 = x, x
        # y = x
        with torch.no_grad():
            for l1, l2 in zip(self.layers, self.layers_myconv2d):
                x1 = l1(x1)
                x2 = l2(x2)
                # y = x1
                # print(allclose(x1, x2))
                inf_norm = torch.max(torch.abs(x1 - x2))
                print(l1)
                print(l2)
                print(inf_norm, inf_norm < 1e-3)
                print()

        print()


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))