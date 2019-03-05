# Deep Learning Tutorial

#### by Geoff French


## Table of Contents

- PyData Amsterdam 2017 slides
- Requirements
- Notebooks for the tutorial
- The Python modules


## Slides

This repo has been used to accompany various talks that I have given, one of which was at PyData Amsterdam 2017. The slides for that talk probably refer to Theano and Lasagne; two neural network toolkits that are no longer developed and maintained. You can find these slides on
[Speakdeck](https://speakerdeck.com/britefury/deep-learning-advanced-techniques-tutorial-pydata-amsterdam-2017).



## Requirements

I recommend using the [Anaconda](http://www.continuum.io/downloads) Python distribution, using Python 3.6.
Anaconda will provide Python, Numpy, Matplotlib and [Scikit-image](http://github.com/scikit-image/scikit-image).

You will need to have the following installed:

- [PyTorch and torchvision](http://pytorch.org) - get version 1.0.
- OpenCV - use `conda` to install it on Linux. On windows go to [Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/), download the `.whl` (wheel) file and install it using `pip install <path_of_opencv_whl>`.



## Notebooks for the tutorial

### Using a pre-trained VGG network

Using a pre-trained network to classify an image:

[Using a pretrained conv-net - VGG net.ipynb](TUTORIAL 01 - Using a pretrained conv-net - VGG net.ipynb)

Converting a network for use in a convolutional fashion:

[Using a pretrained VGG-19 conv-net to find a peacock.ipynb](TUTORIAL 02 - Using a pretrained VGG-19 conv-net to find a peacock.ipynb)



### Saliency

Saliency tells you which parts of the image had most influence over a network's prediction.
Two approaches are demonstrated: [region level saliency](TUTORIAL 03 - Image region-level saliency using VGG-19 conv-net.ipynb)
and [pixel level saliency](TUTORIAL 04 - Image pixel-level saliency using VGG-19 conv-net.ipynb)


### Transfer learning

Transfer learning is the process by which we adapt a pre-trained neural network for other uses. To demonstrate it we train a classifier on the [Kaggle Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats) dataset. There are 3 notebooks that demonstrate the effect of using transfer learning and data augmentation:

[Dogs vs cats with standard learning.ipynb](TUTORIAL 05 - Dogs vs cats with standard learning.ipynb) - train a neural network *without* transfer learning; gets around 10% error rate.

[Dogs vs cats with transfer learning.ipynb](TUTORIAL 05 - Dogs vs cats with transfer learning.ipynb) - train a neural network *with* transfer learning; gets 5-6% error rate

[Dogs vs cats with transfer learning and data augmentation.ipynb](TUTORIAL 05 - Dogs vs cats with transfer learning and data augmentation.ipynb) - train a neural network with transfer learning and data augmentation; gets 2.7-3.7% error rate





## The Python modules

A brief description of the Python modules in this repo:

`utils.py` - utility functions - main converting images for use with pre-trained models

`imagenet_classes.py` - provides a dictionary if ImageNet class names, so that we can give meaningful names to the predictions of a pre-trained network.
