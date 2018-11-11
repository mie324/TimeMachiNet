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


from model import *



gc.collect()



if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument('--mode', choices=['train', 'test'], default='train')

	# Arguments for Training
	parser.add_argument('--epochs', default=1, type=int)
	parser.add_argument('--batch-size', dest='batch_size', default=64, type=int)
    parser.add_argument('--weight-decay', default=1e-5, type=float)
    parser.add_argument('--learning-rate', dest='learning_rate', default=2e-4, type=float)
    parser.add_argument('--b1', dest='b1', default=0.5, type=float)
    parser.add_argument('--b2', dest='b2', default=0.999, type=float)
    parser.add_argument('--shouldplot', dest='sp', default=False, type=bool)

    # Arguments for Testing
    parser.add_argument('--age', required=False, type=int)
    parser.add_argument('--gender', required=False, type=int)
    parser.add_argument('--watermark', action='store_true')

    # General Arguments
    parser.add_argument('--load', required=False, default=None, help='Trained models path for pre-training or for testing')
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default='')
    parser.add_argument('-z', dest='z_channels', default=50, type=int, help='Length of Z vector')
    args = parser.parse_args()

    consts.NUM_Z_CHANNELS = args.z_channels
    net = model.Network()


    if args.mode == 'train':

    	# Initialize
    	if args.load is None:
    		b = (args.b1, args.b2)
    		weight_decay = args.weight_decay
    		lr = args.learning_rate
    	else:
    		b = None
    		weight_decay = None
    		lr = None
    		net.load(args.load)
    		print("hi")

    	data_set = args.input
    	print("Data folder loaded lit")
    	output_dest = args.output
    	os.makedirs(output_dest, exist_ok=True)  # if there isn't already a directory, make a new one
    	print("Output folder is lit")

    	net.train(

    		)

    elif args.mode == 'test':

    	if args.load is None:
    		raise RuntimeError("yo common give us some trained models")

    	net.load(path=args.load, slim=True)

    	output_dest = args.output

    	if not os.path.isdir(output_dest):  # if there isn't already a directory, make a new one
    		os.makedirs(output_dest)

    	image_tensor = tensor_transform(loader(args.input)).to(net.device)
    	net.test_image(
    		image_tensor=image_tensor,
    		age=args.age,
    		gender=args.gender,
    		target=output_dest,
    		watermark=args.watermark
    	)