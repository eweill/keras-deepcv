"""
Train a LeNet model with the MNIST data set

Print model:
	python lenet_mnist.py --print_model

Train and save model:
	python lenet_mnist.py --train_model --epochs 10 --save_model

Train with pretrained weights:
	python lenet_mnist.py --train_model --epochs 10 \
		--save_model --weights data/lenet_mnist.hdf5
"""
import sys
sys.path.append("..")

# Import model architecture and data
from models.classification import lenet
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils

# Import other necessary packages
import numpy as np
import argparse, cv2
import matplotlib.pyplot as plt

def get_mnist():
	"""
	Get the MNIST dataset.
	
	Will download dataset if first time and will be downloaded
	to ~/.keras/datasets/mnist.npz
	Parameters:
		None
	Returns:
		train_data - training data split
		train_labels - training labels
		test_data - test data split
		test_labels - test labels
	"""
	print('[INFO] Loading the MNIST dataset...')
	(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

	# Reshape the data from (samples, height, width) to
	# (samples, height, width, depth) where depth is 1 channel (grayscale)
	train_data = train_data[:, :, :, np.newaxis]
	test_data = test_data[:, :, :, np.newaxis]

	# Normalize the data
	train_data = train_data / 255.0
	test_data = test_data / 255.0

	# Transform labels to one hot labels
	# Example: '0' will become [1, 0, 0, 0, 0, 0, 0, 0, 0]
	#          '1' will become [0, 1, 0, 0, 0, 0, 0, 0, 0]
	#          and so on...
	train_labels = np_utils.to_categorical(train_labels, 10)
	test_labels = np_utils.to_categorical(test_labels, 10)

	return train_data, train_labels, test_data, test_labels

def draw_training_curve(history):
	"""
	Draw training curve

	Parameters:
		history - contains loss and accuracy from training
	Returns:
		None
	"""
	plt.figure(1)

	# History for accuracy
	plt.subplot(211)
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')

	# History for loss
	plt.subplot(212)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')

	plt.show()

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
		action='store_true')
	optional.add_argument('-w', '--weights',
		dest='weights',
		help='Path to weights (hdf5) file',
		default='data/lenet_mnist.hdf5')
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
	if not args.save_weights:
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
	train_data, train_labels, test_data, test_labels = get_mnist()

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
		draw_training_curve(history)

	if args.save_weights:
		print('[INFO] Saving the model weights to file...')
		model.save_weights(args.weights, overwrite=True)
