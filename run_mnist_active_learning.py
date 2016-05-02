#!/usr/bin/env python

from __future__ import print_function

import csv
import argparse

import numpy as np

import mnist_dataset, mnist_architecture, active_learning, cmdline_helpers, utils


def main(model='lenet', num_epochs=500, min_epochs=100, improve_epochs=50,
         subset_sizes=None, validation_intervals=1,
         batchsize=500,
         sample_chooser=None, refine=False, out_path=None,
         csv_path=None, indices_out_path=None):
    if subset_sizes is None:
        subset_sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

    subset_sizes = cmdline_helpers.coerce_num_list_parameter(subset_sizes,
                                                             num_type=int, name='subset_sizes')
    num_epochs = cmdline_helpers.coerce_num_list_parameter(num_epochs, N=len(subset_sizes),
                                                                  num_type=int, name='num_epochs')
    min_epochs = cmdline_helpers.coerce_num_list_parameter(min_epochs, N=len(subset_sizes),
                                                                  num_type=int, name='min_epochs')
    improve_epochs = cmdline_helpers.coerce_num_list_parameter(improve_epochs, N=len(subset_sizes),
                                                                  num_type=int, name='improve_epochs')
    validation_intervals = cmdline_helpers.coerce_num_list_parameter(validation_intervals, N=len(subset_sizes),
                                                                  num_type=int, name='validation_intervals')

    N_train, N_val, N_test = mnist_dataset.train_val_test_size()

    builder = mnist_architecture.network_builder(model)

    mnist = mnist_dataset.MNISTTrainValTest()

    trainer, indices_labelled_history, validation_error_history, test_error_history = \
        active_learning.active_learning_image_classifier(sample_chooser=sample_chooser, model_builder=builder, N_train=N_train,
                                                         batchsize=batchsize,
                                                         refine=refine, datasets_fn=mnist.datasets, subset_sizes=subset_sizes,
                                                         num_epochs=num_epochs,
                                                         min_epochs=min_epochs, improve_epochs=improve_epochs,
                                                         validation_intervals=validation_intervals,
                                                         batch_xform_fn=mnist_dataset.xform_mnist_batch,
                                                         n_train_repetitions_in_case_of_failure=3)

    print('Results:')
    print('N-train\t\tErr')
    for labelled_indices, err in zip(indices_labelled_history, test_error_history):
        print('{0}\t\t{1:.2f}%'.format(labelled_indices.shape[0], err * 100.0))

    if csv_path is not None:
        writer = csv.writer(open(csv_path, 'wb'))
        writer.writerow(['# samples', 'Error %'])
        for labelled_indices, err in zip(indices_labelled_history, test_error_history):
            writer.writerow([labelled_indices.shape[0], err * 100.0])

    if out_path is not None:
        print('Saving model to {0} ...'.format(out_path))
        utils.save_model(out_path, trainer.network)

    if indices_out_path is not None:
        np.save(indices_out_path, indices_labelled_history[-1])



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
    ap.add_argument('--epochs', type=str, default='2000', help='maximum number of epochs to train for')
    ap.add_argument('--min_epochs', type=str, default='100', help='minimum number of epochs to train for, can be a list')
    ap.add_argument('--improve_epochs', type=str, default='500,250,250,250,250,250,250,250,250,250',
                    help='if no improvement in validation error detected after this number of epochs, '
                         'stop training, can be a list')
    ap.add_argument('--subset_sizes', type=str, default='500,1000,1500,2000,2500,3000,3500,4000,4500,5000',
                    help='the sequence of subset sizes that define how the dataset grows, default = 500,100,..,5000')
    ap.add_argument('--val_intervals', type=str, default='50,25,16,12,10,8,7,6,6,5',
                    help='the sequence of validation intervals that define how often the validation score should be computed, '
                         'each interval is a number of epochs, default = 50,25,16,...5; geometric scaling')
    ap.add_argument('--batchsize', type=int, default=500, help='batch size, default=500')
    ap.add_argument('--chooser', type=str, default='confidence:3.0', help='sample chooser: '
                    'order - in order chooser; '
                    'random - random order chooser; '
                    'confidence[:temperature] - choose by confidence, temperature is the (optional) softmax temperature; '
                    'uncertainty - choose by uncertainty; '
                    'uncertainty_cluster[:layer_index[,n_samples]] - choose by uncertainty and clustering; '
                    )
    ap.add_argument('--refine', action='store_true', help='refine existing model when adding new data rather than train new')
    ap.add_argument('--out_path', type=str, default=None, help='path to write the model to when training is complete')
    ap.add_argument('--csv_path', type=str, default=None, help='path for CSV file to write the training results')
    ap.add_argument('--indices_out_path', type=str, default=None, help='path to write the labelled sample indices to to when training is complete')
    args = ap.parse_args()

    chooser = active_learning.make_chooser(args.chooser)

    main(model=args.model, num_epochs=args.epochs,
         min_epochs=args.min_epochs, improve_epochs=args.improve_epochs,
         subset_sizes=args.subset_sizes, validation_intervals=args.val_intervals,
         batchsize=args.batchsize,
         sample_chooser=chooser, refine=args.refine,
         out_path=args.out_path, csv_path=args.csv_path, indices_out_path=args.indices_out_path)
