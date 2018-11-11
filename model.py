"""
This will contain our model
"""

from utils import *
import consts

import logging
import random
from collections import OrderedDict
import cv2
import imageio
from PIL import Image

import torch
import torch.nn as nn
from torch.nn.functional import *
from torch.optim import Adam
from torch.utils.data import DataLoader


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        num_conv_layers = 6

        self.conv_layers = nn.ModuleList()

        def add_conv(layer_list, name, input, output, kernel_size, stride, activation_fcn):
            return layer_list.add_module(
                name,
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=input,
                        out_channels=output,
                        kernel_size=kernel_size,
                        stride=stride,
                    ),
                    activation_fcn
                )
            )
        add_conv(self.conv_layers, 'e_conv_1', input=3, output=64, kernel_size=2, stride=2, activation_fcn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_2', input=64, output=128, kernel_size=2, stride=2, activation_fcn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_3', input=128, output=256, kernel_size=2, stride=2, activation_fcn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_4', input=256, output=512, kernel_size=2, stride=2, activation_fcn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_5', input=512, output=1024, kernel_size=2, stride=2, activation_fcn=nn.ReLU())

        self.fc_1 = nn.Sequential(OrderedDict([
            ('e_fc_1', nn.Linear(
                in_features=1024 * 4 * 4,
                out_features=consts.NUM_Z_CHANNELS
            )),
            ('tanh_1', nn.Tanh())
        ]))

    def forward(self, face):
        x = face
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        return x


class DiscriminatorVec(nn.Module):
    def __init__(self):
        super(DiscriminatorVec, self).__init__()
        dims = (consts.NUM_Z_CHANNELS, consts.NUM_ENCODER_CHANNELS, consts.NUM_ENCODER_CHANNELS // 2,
                consts.NUM_ENCODER_CHANNELS // 4)
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]), 1):
            self.layers.add_module(
                'dvec_fc_%d' % i,
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU()
                )
            )

        self.layers.add_module(
            'dvec_fc_%d' % (i + 1),
            nn.Sequential(
                nn.Linear(out_dim, 1),
            )
        )

    def forward(self, vector):
        output = vector
        for layer in self.layers:
            output = layer(output)
        return output


class DiscriminatorImg(nn.Module):
    def __init__(self):
        pass

    def forward(self, images, labels, device):
        pass


class Generator(nn.Module):
    def __init__(self):
        pass

    def forward(self, vector, age=None, gender=None):
        pass

    def add_deconv(name, input_dim, output_dim, kernel_size, stride, activation_fcn):
        pass

    def decomp(self, x):
        pass


class Network(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __repr__(self):
        pass

    def test_image(self, image_tensor, age, gender, target, watermark):
        pass

    def train(self):
        pass

    def load(self):
        pass
