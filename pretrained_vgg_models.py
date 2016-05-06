# Code taken from:
# https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py
#
# VGG-16 16-layer model and VGG-19 19-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
# License: non-commercial use only

# Download pretrained weights from:
# http://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl
# and
# http://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl

import sys, os, pickle
import numpy as np
import lasagne
import skimage.transform, skimage.color
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer, Pool2DLayer, Conv2DLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

PARAMS_DIR = 'pretrained_models'
VGG16_PATH = os.path.join(PARAMS_DIR, 'vgg16.pkl')
VGG19_PATH = os.path.join(PARAMS_DIR, 'vgg19.pkl')


def download(path, source_url):
    if not os.path.exists(PARAMS_DIR):
        os.makedirs(PARAMS_DIR)
    if not os.path.exists(path):
        print('Downloading {0} to {1}'.format(source_url, path))
        urlretrieve(source_url, path)
    return path


class VGGModel (object):
    """
    VGG model base class

    Override the `build_network` class method to define the network architecture
    """
    def __init__(self, mean_value, class_names, model_name, param_values):
        self.network, self.final_layer_name = self.build_network()
        self.mean_value = mean_value
        self.class_names = class_names
        self.model_name = model_name
        lasagne.layers.set_all_param_values(self.network[self.final_layer_name], param_values)


    @classmethod
    def build_network(cls):
        raise NotImplementedError('Abstract for type {}'.format(cls))

    
    @property
    def final_layer(self):
        return self.network[self.final_layer_name]


    def prepare_image(self, im, image_size=224):
        """
        Prepare an image for classification with VGG; scale and crop to `image_size` x `image_size`.
        Convert RGB channel order to BGR.
        Subtract mean value.

        :param im: input RGB image as numpy array (height, width, channel)
        :param image_size: output image size, default=224. If `None`, scaling and cropping will not be done.
        :return: (raw_image, vgg_image) where `raw_image` is the scaled and cropped image with dtype=uint8 and
            `vgg_image` is the image with BGR channel order and axes (sample, channel, height, width).
        """
        # If the image is greyscale, convert it to RGB
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.repeat(im, 3, axis=2)

        if image_size is not None:
            # Scale the image so that its smallest dimension is the desired size
            h, w, _ = im.shape
            if h < w:
                if h != image_size:
                    im = skimage.transform.resize(im, (image_size, w * image_size / h), preserve_range=True)
            else:
                if w != image_size:
                    im = skimage.transform.resize(im, (h * image_size / w, image_size), preserve_range=True)

            # Crop the central `image_size` x `image_size` region of the image
            h, w, _ = im.shape
            im = im[h//2 - image_size // 2:h // 2 + image_size // 2, w // 2 - image_size // 2:w // 2 + image_size // 2]

        # Convert to uint8 type
        rawim = np.copy(im).astype('uint8')

        # Images come in RGB channel order, while VGG net expects BGR:
        im = im[:, :, ::-1]

        # Subtract the mean
        im = im - self.mean_value

        # Shuffle axes from (height, width, channel) to (channel, height, width)
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

        # Add the sample axis to the image; (channel, height, width) -> (sample, channel, height, width)
        im = im[np.newaxis]

        return rawim, floatX(im)


    def inv_prepare_image(self, image):
        """
        Perform the inverse of `prepare_image`; usually used to display an image prepared for classification
        using a VGG net.

        :param im: the image to process
        :return: processed image
        """
        if len(image.shape) == 4:
            # We have a sample dimension; can collapse it if there is only 1 sample
            if image.shape[0] == 1:
                image = image[0]
            else:
                raise ValueError('Sample dimension has > 1 samples ({})'.format(image.shape[0]))

        # Move the channel axis: (C, H, W) -> (H, W, C)
        image = np.rollaxis(image, 0, 3)
        # Add the mean
        image = image + self.mean_value
        # Clip to [0,255] range
        image = image.clip(0.0, 255.0)
        # Convert to uint8 type
        image = image.astype('uint8')
        # Flip channel order BGR to RGB
        image = image[:,:,::-1]
        return image


    @classmethod
    def from_loaded_params(cls, loaded_params):
        """
        Construct a model given parameters loaded from a pickled VGG model
        :param loaded_params: a dictionary resulting from loading a pickled VGG model
        :return: the model
        """
        return cls(loaded_params['mean value'], loaded_params['synset words'], loaded_params['model name'],
                   loaded_params['param values'])

    @staticmethod
    def unpickle_from_path(path):
        # Oh... the joys of Py2 vs Py3
        with open(path, 'rb') as f:
            if sys.version_info[0] == 2:
                return pickle.load(f)
            else:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                return u.load()


class VGG16Model (VGGModel):
    @classmethod
    def build_network(cls):
        net = {}
        # Input layer: shape is of the form (sample, channel, height, width).
        # We are using 3 channel images of size 224 x 224.
        # We leave the sample dimension with no size (`None`) so that the
        # minibatch size is whatever we need it to be when we use it
        net['input'] = InputLayer((None, 3, 224, 224))

        # First two convolutional layers: 64 filters, 3x3 convolution, 1 pixel padding
        net['conv1_1'] = Conv2DLayer(net['input'], 64, 3, pad=1, flip_filters=False)
        net['conv1_2'] = Conv2DLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
        # 2x2 max-pooling; will reduce size from 224x224 to 112x112
        net['pool1'] = Pool2DLayer(net['conv1_2'], 2)

        # Two convolutional layers, 128 filters
        net['conv2_1'] = Conv2DLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
        net['conv2_2'] = Conv2DLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
        # 2x2 max-pooling; will reduce size from 112x112 to 56x56
        net['pool2'] = Pool2DLayer(net['conv2_2'], 2)

        # Three convolutional layers, 256 filters
        net['conv3_1'] = Conv2DLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_2'] = Conv2DLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
        net['conv3_3'] = Conv2DLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
        # 2x2 max-pooling; will reduce size from 56x56 to 28x28
        net['pool3'] = Pool2DLayer(net['conv3_3'], 2)

        # Three convolutional layers, 512 filters
        net['conv4_1'] = Conv2DLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
        net['conv4_2'] = Conv2DLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
        net['conv4_3'] = Conv2DLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
        # 2x2 max-pooling; will reduce size from 28x28 to 14x14
        net['pool4'] = Pool2DLayer(net['conv4_3'], 2)

        # Three convolutional layers, 512 filters
        net['conv5_1'] = Conv2DLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
        net['conv5_2'] = Conv2DLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
        net['conv5_3'] = Conv2DLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
        # 2x2 max-pooling; will reduce size from 14x14 to 7x7
        net['pool5'] = Pool2DLayer(net['conv5_3'], 2)

        # Dense layer, 4096 units
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        # 50% dropout (only applied during training, turned off during prediction)
        net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)

        # Dense layer, 4096 units
        net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
        # 50% dropout (only applied during training, turned off during prediction)
        net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)

        # Final dense layer, 1000 units: 1 for each class
        net['fc8'] = DenseLayer(net['fc7_dropout'], num_units=1000, nonlinearity=None)
        # Softmax non-linearity that will generate probabilities
        net['prob'] = NonlinearityLayer(net['fc8'], softmax)

        return net, 'prob'

    @staticmethod
    def load_params():
        download(VGG16_PATH, 'http://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl')
        return VGGModel.unpickle_from_path(VGG16_PATH)

    @classmethod
    def load(cls):
        return cls.from_loaded_params(cls.load_params())


class VGG19Model (VGGModel):
    @classmethod
    def build_network(cls):
        net = {}
        # Input layer: shape is of the form (sample, channel, height, width).
        # We are using 3 channel images of size 224 x 224.
        # We leave the sample dimension with no size (`None`) so that the
        # minibatch size is whatever we need it to be when we use it
        net['input'] = InputLayer((None, 3, 224, 224))

        # First two convolutional layers: 64 filters, 3x3 convolution, 1 pixel padding
        net['conv1_1'] = Conv2DLayer(net['input'], 64, 3, pad=1, flip_filters=False)
        net['conv1_2'] = Conv2DLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
        # 2x2 max-pooling; will reduce size from 224x224 to 112x112
        net['pool1'] = Pool2DLayer(net['conv1_2'], 2)

        # Two convolutional layers, 128 filters
        net['conv2_1'] = Conv2DLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
        net['conv2_2'] = Conv2DLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
        # 2x2 max-pooling; will reduce size from 112x112 to 56x56
        net['pool2'] = Pool2DLayer(net['conv2_2'], 2)

        # Four convolutional layers, 256 filters
        net['conv3_1'] = Conv2DLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_2'] = Conv2DLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
        net['conv3_3'] = Conv2DLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_4'] = Conv2DLayer(net['conv3_3'], 256, 3, pad=1, flip_filters=False)
        # 2x2 max-pooling; will reduce size from 56x56 to 28x28
        net['pool3'] = Pool2DLayer(net['conv3_4'], 2)

        # Four convolutional layers, 512 filters
        net['conv4_1'] = Conv2DLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
        net['conv4_2'] = Conv2DLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
        net['conv4_3'] = Conv2DLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
        net['conv4_4'] = Conv2DLayer(net['conv4_3'], 512, 3, pad=1, flip_filters=False)
        # 2x2 max-pooling; will reduce size from 28x28 to 14x14
        net['pool4'] = Pool2DLayer(net['conv4_4'], 2)

        # Four convolutional layers, 512 filters
        net['conv5_1'] = Conv2DLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
        net['conv5_2'] = Conv2DLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
        net['conv5_3'] = Conv2DLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
        net['conv5_4'] = Conv2DLayer(net['conv5_3'], 512, 3, pad=1, flip_filters=False)
        # 2x2 max-pooling; will reduce size from 14x14 to 7x7
        net['pool5'] = Pool2DLayer(net['conv5_4'], 2)

        # Dense layer, 4096 units
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        # 50% dropout (only applied during training, turned off during prediction)
        net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)

        # Dense layer, 4096 units
        net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
        # 50% dropout (only applied during training, turned off during prediction)
        net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)

        # Final dense layer, 1000 units: 1 for each class
        net['fc8'] = DenseLayer(net['fc7_dropout'], num_units=1000, nonlinearity=None)
        # Softmax non-linearity that will generate probabilities
        net['prob'] = NonlinearityLayer(net['fc8'], softmax)

        return net, 'prob'

    @staticmethod
    def load_params():
        download(VGG19_PATH, 'http://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl')
        return VGGModel.unpickle_from_path(VGG19_PATH)

    @classmethod
    def load(cls):
        return cls.from_loaded_params(cls.load_params())
