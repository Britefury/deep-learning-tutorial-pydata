# Deep Learning Tutorial

#### for PyData London 2016, by Geoffrey French


## Table of Contents

- Requirements
- Downloading datasets
- Notebooks for the tutorial
- The Python modules


## Requirements

I recommend using the [Anaconda](http://www.continuum.io/downloads) Python distribution.
For this tutorial we are using Python 2. Anaconda will provide Python, Numpy, Matplotlib
and [Scikit-image](http://github.com/scikit-image/scikit-image). That are needed.

You will need to have the following installed:

- [Theano](http://github.com/Theano/Theano) - get version 0.8. While `pip install Theano` should do it,
I checked out the `0.8.X` branch from Github.
- [Lasagne](http://github.com/Lasagne/Lasagne)
- [Fuel](http://github.com/mila-udem/fuel) - get version 0.2

To install Theano and Lasagne, you could follow the instructions for Lasagne that suggest:
```
pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
```

## Downloading datasets

After installing the libraries mentioned above, you will need to download the datasets using fuel.
Create a directory into which fuel should store its data, say `/home/me/fuel_data`. Now create
the file `/home/me/.fuelrc` with the following contents:

```
# ~/.fuelrc
data_path: "/home/me/fuel_data/"
```

Now, from the command line, go into the `/home/me/fuel_data` directory and run:

```fuel-download mnist```

Then:

```fuel-convert mnist```

You can also download CIFAR-10:

```
fuel-download cifar10
fuel-convert cifar10
```

and SVHN (the extra argument identifies the type of SVHN we are downloading):

```
fuel download svhn 2
fuel-convert svhn 2
```


## Notebooks for the tutorial

### Theano and Lasagne basics

These notebooks give a brief introduction to Theano and Lasagne:

[Theano basics](Theano basics.ipynb) and [Lasagne basics](Lasagne basics.ipynb)


### Using a pre-trained VGG network

Using a pre-trained network to classify an image:

[Using a pretrained conv-net - VGG net.ipynb](Using a pretrained conv-net - VGG net.ipynb)

Converting a network for use in a convolutional fashion:

[Using a pretrained VGG-19 conv-net to find a peacock.ipynb](Using a pretrained VGG-19 conv-net to find a peacock.ipynb)



### Saliency

Saliency tells you which parts of the image had most influence over a network's prediction.
Two approaches are demonstrated: [region level saliency](Image region-level saliency using VGG-19 conv-net.ipynb)
and [pixel level saliency](Image pixel-level saliency using VGG-19 conv-net.ipynb)


### Just for fun

To generate Deep Dreams - images with a hallucinogenic appearance - we perform gradient descent or ascent
on an image rather than the network weights: [Deep Dreams](Deep dreams using VGG-19 conv-net.ipynb).



## The Python modules

A brief description of the Python modules in this repo:

`cmdline_helpers.py` - a couple of helper functions for proessing some command line arguments.

`utils.py` - utility functions, including loading and saving networks to disk

`trainer.py` - the `Trainer` class provides a flexible neural network training loop, with
support for early termination, results monitoring, etc. Implemented here to save you having
to implement it elsewhere.

`image_classifier.py` - the `ImageClassifier` class implements a basic image classification
model, with support for training, prediction, etc. You provide a function to build the network
of the architecture that you want and the training data, the rest is hanled for you.

`active_learning.py` - functions that implement active learning, in which the size of a dataset is
incrementally increased by choosing samples for labelling that will be most helpful.

`mnist_architecture.py` - provides a variety of network architectures for use on the MNIST
hand-written digits dataset

`mnist_dataset.py` - uses the Fuel library to provide convenient access to the MNIST
hand-written digits dataset

`pretrained_vgg_models.py` - functions for downloading the VGG-16 and VGG-19 pre-trained
convolutional network models, along with code that will construct the network architecture
using [Lasagne](http://github.com/Lasagne/Lasagne).

`run_mnist.py` - train and MNIST digit classifier

`run_mnist_active_learning.py` - use active learning to gradually grow a dataset that is
a subset of MNIST.

