"""
Train an VGG16/VGG19 model with the MNIST/CIFAR10 dataset

VGG16
Print model:
	python vgg_train.py --net vgg16 --print_model

VGG19


Print model:
	python alexnet_train.py --print_model --dataset mnist

Train and save model:
	python alexnet_train.py --train_model --epochs 10 \
		--save_weights data/alexnet_mnist_trained.hdf5 \
		--dataset cifar10

Train with pretrained weights
	python alexnet_train.py --train_model --epochs 10 \
		--save_weights data/alexnet_mnist_trained.hdf5 \
		--weights data/alexnet_mnist.hdf5 \
		--dataset cifar10
"""

import sys
sys.path.append("..")

# Import model architecture and data
from models.classification import vgg
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import rmsprop
from datasets import cifar10
from datasets import mnist
from utils import draw

# Import other necessary packages
import numpy as np
import argparse, cv2, os
import matplotlib.pyplot as plt


def parse_args():
	"""
	Parse command line arguments.

	Parameters:
		None
	Returns:
		parser arguments
	"""
	parser = argparse.ArgumentParser(description='LeNet model')
	optional = parser._action_groups.pop()
	required = parser.add_argument_group('required arguments')
	required.add_argument('--net',
		dest='net',
		help='Choice of network architecture',
		choices=['vgg16', 'vgg19'])
	optional.add_argument('--dataset',
		dest='dataset',
		help='Choice of dataset to train model',
		choices=[None, 'mnist', 'cifar10'],
		default=None)
	optional.add_argument('--print_model',
		dest='print_model',
		help='Print LeNet model',
		action='store_true')
	optional.add_argument('--train_model',
		dest='train_model',
		help='Train LeNet on MNIST',
		action='store_true')
	optional.add_argument('-s', '--save_weights',
		dest='save_weights',
		help='Save the trained weights',
		default=None)
	optional.add_argument('-w', '--weights',
		dest='weights',
		help='Path to weights (hdf5) file',
		default=None)
	optional.add_argument('-e', '--epochs',
		dest='epochs',
		help='Number of epochs for training',
		type=int,
		default=20)
	optional.add_argument('--data_augmentation',
		dest='data_augmentation',
		help='Use data augmentations for input',
		action='store_true')
	optional.add_argument('--viz_training',
		dest='viz_training',
		help='Visualize the training curve',
		action='store_true')
	parser._action_groups.append(optional)
	return parser.parse_args()

if __name__ == '__main__':
	# Command line parameters
	args = parse_args()

	if args.net == None:
		print('Please add a network model...')
		exit(0)

	# Construct VGG model
	if args.net == 'vgg16':
		if args.weights is None:
			if args.dataset == 'mnist':
				model = vgg.vgg16_model(img_shape=(28, 28, 1))
			elif args.dataset == 'cifar10':
				model = vgg.vgg16_model(img_shape=(32, 32, 3))
			else:
				model = vgg.vgg16_model()
		else:
			if args.dataset == 'mnist':
				model = vgg.vgg16_model(img_shape=(28, 28, 1),
						weights=args.weights)
			elif args.dataset == 'cifar10':
				model = vgg.vgg16_model(img_shape=(32, 32, 3),
						weights=args.weights)
			else:
				model = vgg.vgg16_model(weights=args.weights)
	elif args.net == 'vgg19':
		if args.weights is None:
			if args.dataset == 'mnist':
				model = vgg.vgg19_model(img_shape=(28, 28, 1))
			elif args.dataset == 'cifar10':
				model = vgg.vgg19_model(img_shape=(32, 32, 3))
			else:
				model = vgg.vgg16_model()
		else:
			if args.dataset == 'mnist':
				model = vgg.vgg19_model(img_shape=(28, 28, 1),
						weights=args.weights)
			elif args.dataset == 'cifar10':
				model = vgg.vgg19_model(img_shape=(32, 32, 3),
						weights=args.weights)
			else:
				model = vgg.vgg16_model(weights=args.weights)

	# Print model summary
	if args.print_model:
		model.summary()