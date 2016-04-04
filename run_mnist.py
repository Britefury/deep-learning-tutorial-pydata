#!/usr/bin/env python

from __future__ import print_function

import argparse

import numpy as np

import mnist_dataset, mnist_architecture, image_classifier, trainer, utils


def main(model='lenet', num_epochs=500, min_epochs=200, improve_epochs=250, batchsize=500,
         training_subset=None, aug_factor=1, out_path=None):
    # Load the dataset
    print("Loading data...")
    mnist = mnist_dataset.MNISTTrainValTest()

    # Generate the indices of the subset of the training set
    training_subset_indices = None
    if training_subset is not None:
        training_subset_indices = mnist.balanced_train_subset_indices(training_subset)

    # Get the train, validation and test sets
    train_ds, val_ds, test_ds = mnist.datasets(training_subset_indices)

    # Get network builder function for named model
    builder = mnist_architecture.network_builder(model)

    # Build the image classifier for the given model builder
    clf = image_classifier.ImageClassifier.for_model(builder)

    # Set verbosity
    clf.trainer.report(verbosity=trainer.VERBOSITY_EPOCH)

    # Set data transformation function
    clf.trainer.data_xform_fn(batch_xform_fn=mnist_dataset.xform_mnist_batch)

    # Set training length
    clf.trainer.train_for(num_epochs=num_epochs, min_epochs=min_epochs, val_improve_num_epochs=improve_epochs,
                          val_improve_epochs_factor=0)

    # Train
    clf.trainer.train(train_ds, val_ds, test_ds, batchsize=batchsize)

    if out_path is not None:
        print('Saving model to {0} ...'.format(out_path))
        utils.save_model(out_path, clf.network)


if __name__ == '__main__':
    ap = argparse.ArgumentParser('MNIST neural net trainer')
    ap.add_argument('--model', type=str, default='dcnn',
                    help="Model type; "
                    "'logit' for a logistic regression model, "
                    "'mlp64' for a Multi-Layer Perceptron (MLP) with a single 64 unit layer, "
                    "'mlp256_256' for a Multi-Layer Perceptron (MLP) with two 256 unit layers, "
                    "'lenet' for a convolutional network based on the LeNet architecture (5c20-p2-5c50-p2-fc256-fc10), "
                    "'cnn533' for a CNN with 5c32-p2-3c32-3c32-p2-fc256-fc10 architecture, "
                    "'dcnn' for a CNN with 3c32-3c32-p2-3c32-3c32-p2-fc256-fc10 architecture")
    ap.add_argument('--epochs', type=int, default=500, help='number of epochs to train for')
    ap.add_argument('--min_epochs', type=int, default=200, help='minimum number of epochs to train for')
    ap.add_argument('--improve_epochs', type=int, default=200, help='if no improvement in validation error detected after this number of epochs, stop training')
    ap.add_argument('--batchsize', type=int, default=500, help='mini-batch size')
    ap.add_argument('--subset', type=int, default=None, help='number of training examples to use, default=use all')
    ap.add_argument('--out_path', type=str, default=None, help='path to write the model to when training is complete')
    args = ap.parse_args()

    main(model=args.model, num_epochs=args.epochs, min_epochs=args.min_epochs, improve_epochs=args.improve_epochs,
         batchsize=args.batchsize, training_subset=args.subset,
         out_path=args.out_path)
