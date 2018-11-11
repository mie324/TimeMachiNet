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
		pass

	def add_conv(layer_list, name, input, output, kernel_size, stride, padding, activation_fcn):
		pass

	def forward(self, face):
		pass

class DiscriminatorVec(nn.Module):
	def __init__(self):
		pass

	def forward(self, vector):
		pass

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
	