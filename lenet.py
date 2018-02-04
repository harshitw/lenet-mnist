## implement Convolutional Neural Network using LaNet architecture using keras

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class LeNet:
    @staticmethod
    # weightsPath can be used to load a pre trained model
    def build(width, height, depth, classes, weightsPath = None):
        # initialize the model
        model = Sequential()
        # create first set of CONV => RELU => POOL
        model.add(Convolution2D(20, 5, 5, border_mode = "same", input_shape = (depth, height, width)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        # second set of CONV => RELU => POOL
        model.add(Convolution2D(50, 5, 5, border_mode = "same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        # fully connected layers often called dense layers of lenet architecture
        # set FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes)) # number of class labels i.e. in this case we have 10 classes
        model.add(Activation("softmax")) # multinomial logistic regression that returns a list of probabilities

        # if a weights path is supplied (indicating that the model was pretrained), then load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        # return the constructed network architecture
        return model
