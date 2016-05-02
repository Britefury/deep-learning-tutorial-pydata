# Deep Learning Tutorial

#### for PyData London 2016, by Geoffrey French


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