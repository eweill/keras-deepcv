"""
AlexNet Keras Implementation

BibTeX Citation:

@inproceedings{krizhevsky2012imagenet,
  title={Imagenet classification with deep convolutional neural networks},
  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle={Advances in neural information processing systems},
  pages={1097--1105},
  year={2012}
}
"""

# Import necessary packages
import argparse

# Import necessary components to build LeNet
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def alexnet_model(img_shape=(224, 224, 3), n_classes=1000, l2_reg=0.,
	weights=None):

	# Initialize model
	alexnet = Sequential()

	alexnet.add(Conv2D(64, (11, 11), padding="valid",
		input_shape=img_shape, kernel_regularizer=l2(l2_reg)))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(3, 3)))

	alexnet.add(Conv2D(128, (7, 7), padding="valid",
		kernel_regularizer=l2(l2_reg)))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(3, 3)))

	alexnet.add(Conv2D(192, (3, 3), padding="valid",
		kernel_regularizer=l2(l2_reg)))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(3, 3)))

	alexnet.add(Conv2D(256, (3, 3), padding="valid",
		kernel_regularizer=l2(l2_reg)))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(3, 3)))

	alexnet.add(Flatten())
	alexnet.add(Dense(4096, input_dim=12*12*256))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	
	alexnet.add(Dense(4096, input_dim=4096))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))

	alexnet.add(Dense(n_classes, input_dim=4096))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('softmax'))

	if weights is not None:
		alexnet.load_weights(weights)

	return alexnet

def parse_args():
	"""
	Parse command line arguments.

	Parameters:
		None
	Returns:
		parser arguments
	"""
	parser = argparse.ArgumentParser(description='AlexNet model')
	optional = parser._action_groups.pop()
	required = parser.add_argument_group('required arguments')
	optional.add_argument('--print_model',
		dest='print_model',
		help='Print AlexNet model',
		action='store_true')
	parser._action_groups.append(optional)
	return parser.parse_args()

if __name__ == "__main__":
	# Command line parameters
	args = parse_args()

	# Create LeNet model
	model = alexnet_model()

	# Print
	if args.print_model:
		model.summary()