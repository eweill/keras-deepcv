"""
LeNet Keras Implementation

BibTeX Citation:

@article{lecun1998gradient,
  title={Gradient-based learning applied to document recognition},
  author={LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  journal={Proceedings of the IEEE},
  volume={86},
  number={11},
  pages={2278--2324},
  year={1998},
  publisher={IEEE}
}
"""

# Import necessary packages
import argparse

# Import necessary components to build LeNet
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.regularizers import l2

def lenet_model(img_shape=(28, 28, 1), n_classes=10, l2_reg=0.,
	weights=None):

	# Initialize model
	lenet = Sequential()

	# 2 sets of CRP (Convolution, RELU, Pooling)
	lenet.add(Conv2D(20, (5, 5), padding="same",
		input_shape=img_shape, kernel_regularizer=l2(l2_reg)))
	lenet.add(Activation("relu"))
	lenet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	lenet.add(Conv2D(50, (5, 5), padding="same",
		kernel_regularizer=l2(l2_reg)))
	lenet.add(Activation("relu"))
	lenet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# Fully connected layers (w/ RELU)
	lenet.add(Flatten())
	lenet.add(Dense(500, kernel_regularizer=l2(l2_reg)))
	lenet.add(Activation("relu"))

	# Softmax (for classification)
	lenet.add(Dense(n_classes, kernel_regularizer=l2(l2_reg)))
	lenet.add(Activation("softmax"))

	if weights is not None:
		lenet.load_weights(weights)

	# Return the constructed network
	return lenet

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
	parser._action_groups.append(optional)
	return parser.parse_args()

if __name__ == "__main__":
	# Command line parameters
	args = parse_args()

	# Create LeNet model
	model = lenet_model()

	# Print
	if args.print_model:
		model.summary()