import lasagne


def build_logit(input_var=None):
    """
    Logistic regression

    :param input_var: Theano input variable
    :return: neural network as a Lasagne layer
    """
    # Input layer
    # Shape: (unspecified batch size, 1 channel, 28 rows, 28 columns)
    # Link to the Theano variable `input_var`
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    # Dense output layer, 10 units; the softmax function is
    # provided by the `ImageClassifier` class so we use the
    # identity function here.
    network = lasagne.layers.DenseLayer(network, num_units=10,
            nonlinearity=lasagne.nonlinearities.identity)

    return network


def build_mlp_64(input_var=None):
    """
    Create an MLP with one layer with 64 hidden units

    :param input_var: Theano input variable
    :return: neural network as a Lasagne layer
    """
    # Input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    # Dense layer of 64 units, ReLU non-linearity, init weights
    # with He's scheme
    network = lasagne.layers.DenseLayer(network, num_units=64,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    # 50% dropout
    network = lasagne.layers.DropoutLayer(network, p=0.5)

    # Dense output layer, 10 units; softmax function provided by
    # `ImageClassifier` class so use identity function here.
    network = lasagne.layers.DenseLayer(network, num_units=10,
            nonlinearity=lasagne.nonlinearities.identity)

    return network


def build_mlp_256_256(input_var=None):
    """
    Create an MLP with two layers each with 256 hidden units

    :param input_var: Theano input variable
    :return: neural network as a Lasagne layer
    """
    # Input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    # Dense layer of 256 units, ReLU non-linearity, init weights
    # with He's scheme
    network = lasagne.layers.DenseLayer(network, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    # Dense layer of 256 units, ReLU non-linearity, init weights
    # with He's scheme
    network = lasagne.layers.DenseLayer(network, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    # 50% dropout
    network = lasagne.layers.DropoutLayer(network, p=0.5)

    # Dense output layer, 10 units; softmax function provided by
    # `ImageClassifier` class so use identity function here.
    network = lasagne.layers.DenseLayer(network, num_units=10,
            nonlinearity=lasagne.nonlinearities.identity)

    return network


def build_cnn_lenet(input_var=None):
    """
    Create an CNN that roughly replicates the LeNet architecture;
    2 convolutional layers with 5x5 kernel sizes, wth 20 and 50
    filters respectively followed by a dense layer with 256 units.

    :param input_var: Theano input variable
    :return: neural network as a Lasagne layer
    """
    # Input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    # Convolutional layer with 20 kernels of size 5x5
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=20, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 50 5x5 kernels and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=50, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units followed by 50% dropout
    network = lasagne.layers.DenseLayer(
            network, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network, p=0.5)

    # Dense output layer, 10 units; softmax function provided by
    # `ImageClassifier` class so use identity function here.
    network = lasagne.layers.DenseLayer(network, num_units=10,
            nonlinearity=lasagne.nonlinearities.identity)

    return network


def build_cnn_533(input_var=None):
    """
    Create an CNN with 3 convolutional layers, with kernel size of
    5x5, 3x3 and 3x3, with 32 filters each.

    :param input_var: Theano input variable
    :return: neural network as a Lasagne layer
    """

    # Input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    # Convolutional layer with 32 kernels of size 5x5
    # then 2x2 max pooling
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # 2 convolutional layers, 32 3x3 filters
    # then 2x2 max pooling
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units followed by 50% dropout
    network = lasagne.layers.DenseLayer(
            network, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network, p=0.5)

    # Dense output layer, 10 units; softmax function provided by
    # `ImageClassifier` class so use identity function here.
    network = lasagne.layers.DenseLayer(network, num_units=10,
            nonlinearity=lasagne.nonlinearities.identity)

    return network


def build_deep_cnn(input_var=None):
    """
    Create an CNN with 4 convolutional layers, each with
    32 filters, 3x3 in size

    :param input_var: Theano input variable
    :return: neural network as a Lasagne layer
    """

    # Input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    # 2 convolutional layers, 32 3x3 filters
    # then 2x2 max pooling
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # 2 convolutional layers, 32 3x3 filters
    # then 2x2 max pooling
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units followed by 50% dropout
    network = lasagne.layers.DenseLayer(
            network, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network, p=0.5)

    # Dense output layer, 10 units; softmax function provided by
    # `ImageClassifier` class so use identity function here.
    network = lasagne.layers.DenseLayer(network, num_units=10,
            nonlinearity=lasagne.nonlinearities.identity)

    return network


def network_builder(model):
        # Create neural network model (depending on first command line parameter)
    if model == 'logit':
        return build_logit
    elif model == 'mlp64':
        return build_mlp_64
    elif model == 'mlp256_256':
        return build_mlp_256_256
    elif model == 'lenet':
        return build_cnn_lenet
    elif model == 'cnn533':
        return build_cnn_533
    elif model == 'dcnn':
        return build_deep_cnn
    else:
        raise ValueError("Unrecognized model type %r." % model)

