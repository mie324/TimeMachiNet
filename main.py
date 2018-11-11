"""
This will contain the training loop
"""


import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy

import argparse
import os


from models import *



gc.collect()




