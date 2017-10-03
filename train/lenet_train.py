"""
Train a LeNet model with the MNIST dataset

Print model:
	python lenet_mnist.py --print_model

Train and save model:
	python lenet_mnist.py --train_model --epochs 10 \
		--save_weights data/lenet_mnist_trained.hdf5

Train with pretrained weights:
	python lenet_mnist.py --train_model --epochs 10 \
		--save_weights data/lenet_mnist_trained.hdf5 \
		--weights data/lenet_mnist.hdf5
"""
import sys
sys.path.append("..")

# Import model architecture and data
from models.classification import lenet
from keras.optimizers import SGD
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
	parser._action_groups.append(optional)
	return parser.parse_args()

if __name__ == '__main__':
	# Command line parameters
	args = parse_args()

	# Construct LeNet model
	if args.weights is None:
		model = lenet.lenet_model()
	else:
		model = lenet.lenet_model(weights=args.weights)
	if args.print_model:
		model.summary()

	# Compile model
	model.compile(loss="categorical_crossentropy",
		optimizer=SGD(lr=0.01),
		metrics=["accuracy"])

	# Get MNIST data
	train_data, train_labels, test_data, test_labels = mnist.get_data()

	# Train the model
	if args.train_model:
		print('[INFO] Training the model...')
		history = model.fit(train_data, train_labels,
			batch_size=128,
			epochs=args.epochs,
			validation_data=(test_data, test_labels),
			verbose=1)

		# Evaluate model
		print('[INFO] Evaluating the trained model...')
		(loss, accuracy) = model.evaluate(test_data, test_labels,
			batch_size=128,
			verbose=1)
		print('[INFO] accuracy: {:.2f}%'.format(accuracy * 100))

		# Visualize training history
		draw.draw_training_curve(history)

	# Save model and weights
	if args.save_weights is not None:
		print('[INFO] Saving the model weights to file...')
		if not os.path.exists(os.path.dirname(args.save_weights)):
			os.path.makedirs(os.path.dirname(args.save_weights))
		if os.path.isfile(args.save_weights):
			os.remove(args.save_weights)
		model.save_weights(args.save_weights, overwrite=True)
