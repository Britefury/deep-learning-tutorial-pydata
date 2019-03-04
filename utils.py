import numpy as np
import skimage.util


def prepare_image(im, image_mean, image_std, image_size=224):
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

    # Convert to float type
    im = skimage.util.img_as_float(im)

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
        im = im[h // 2 - image_size // 2:h // 2 + image_size // 2, w // 2 - image_size // 2:w // 2 + image_size // 2]

    rawim = im.copy()

    # Shuffle axes from (height, width, channel) to (channel, height, width)
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Subtract the mean and divide by the std-dev
    # Note that we add two axes to the mean and std-dev for height and width so that they broadcast with the image array
    im = (im - image_mean[:, None, None]) / image_std[:, None, None]

    # Add the sample axis to the image; (channel, height, width) -> (sample, channel, height, width)
    im = im[None, ...]

    return rawim, im.astype(np.float32)


def inv_prepare_image(image, image_mean, image_std):
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
    image = image.transpose(1, 2, 0)
    # Add the mean
    image = image * image_std + image_mean
    # Clip to [0,1] range
    image = image.clip(0.0, 1.0)
    # Convert to uint8 type
    image = skimage.util.img_as_ubyte(image)
    return image
