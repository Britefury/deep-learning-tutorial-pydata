"""
This module provides
"""

import numpy as np

import image_classifier, trainer


class SampleChooser (object):
    """
    Sample chooser abstract base class

    Invoke the `choose_samples` method to choose sames to incorporate into a
    further round of active learning.
    """
    def choose_samples(self, N_samples, train_unlabelled_ds, N_unused, model, batch_xform_fn):
        """
        Choose additional samples for the next round of active learning

        :param N_samples: number of samples to choose
        :param train_unlabelled_ds: dataset containing unused samples from training set
        :param N_unused: number of unused samples
        :param model: neural network model used for choosing additional samples
        :param batch_xform_fn: batch transformation function in case data should be
            transformed for use
        :return: an array of indices identifying the samples to label for the next round
        """
        raise NotImplementedError, 'abstract for type {-}'.format(type(self))


class SampleChooserOrder (SampleChooser):
    """
    Chooses additional samples in order; choose the first `N_samples` samples
    """
    def choose_samples(self, N_samples, train_unlabelled_ds, N_unused, model, batch_xform_fn):
        return np.arange(N_samples)


class SampleChooserRandom (SampleChooser):
    """
    Chooses additional samples randomly; choose the first `N_samples` samples
    """
    def choose_samples(self, N_samples, train_unlabelled_ds, N_unused, model, batch_xform_fn):
        indices = np.arange(N_unused)
        np.random.shuffle(indices)
        return indices[:N_samples]


class SampleChooserConfidence (SampleChooser):
    def __init__(self, temperature):
        self.temperature = temperature

    def choose_samples(self, N_samples, train_unlabelled_ds, N_unused, model, batch_xform_fn):
        # Use the network to predict soft probabilities for the unused training samples
        pred_prob_unused = model.predict_prob(train_unlabelled_ds, temperature=self.temperature,
                                              batch_xform_fn=batch_xform_fn)
        # Compute the confidence of the unused training set predictions
        confidence_unused = np.max(pred_prob_unused, axis=1)
        # Use argsort to get the order of the confidence values in ascending order
        order = np.argsort(confidence_unused)
        # Select the first `N` samples
        return order[:N_samples]


def make_chooser(desc):
    """
    Make sample chooser described by a textual description acquired from the command line.
    :param desc: chooser description
    :return: a `SampleChooser` instance
    """
    if isinstance(desc, str) and ':' in desc:
        name, _, args = desc.partition(':')
        args = args.split(',')
    else:
        name = desc
        args = []
    if name == 'order':
        print('Adding in order')
        return SampleChooserOrder()
    elif name == 'random':
        print('Adding in random order')
        return SampleChooserRandom()
    elif name == 'confidence':
        temperature = 5.0
        if len(args) >= 1:
            try:
                temperature = float(args[0])
            except:
                pass
        print('Adding in inverse confidence order with temperature={0}'.format(temperature))
        return SampleChooserConfidence(temperature)
    else:
        return SampleChooserOrder()


def active_learning_image_classifier(sample_chooser, model_builder, N_train, batchsize,
                                     refine, datasets_fn, subset_sizes,
                                     num_epochs, min_epochs, improve_epochs,
                                     validation_intervals, batch_xform_fn=None,
                                     n_train_repetitions_in_case_of_failure=1):
    """
    Train an image classifier using active learning.

    Active learning reduces the number of training samples required by gradually growing
    the training set in rounds by choosing the most valuable samples from a set of
    unlabelled samples for which ground truths should be provided.

    There are a number of sample choosing strategies, implemented by subclasses of `SampleChooser`.

    :param sample_chooser: an instance of a `SampleChooser` subclass that is used to select
        the samples for the next round
    :param model_builder: model builder function that determines the architecture of the network
    :param N_train: the size of the complete training set
    :param batchsize: minibatch size used during training
    :param refine: if False train a new model each round, if True train a new one
    :param datasets_fn: a function for acquiring the dataset, of the form
        `fn(indices) -> (train_ds, val_ds, test_ds)` where `indices` is an array of indices
        that identifies the subset of samples from the training set to use, `train_ds` is the
        subset of the training set containing samples chosen by `indices`, `val_ds` and `test_ds`
        are the validation and test datasets respectively and are always the same irrespective
        of `indices`
    :param subset_sizes: A list of integers with an item for each active learning round
        that specifies the number of samples in the training (sub)set
    :param num_epochs: A list of integers with an item for each round specifying the number of
        epochs to train for (see `Trainer.train_for(num_epochs)`)
    :param min_epochs: A list of integers with an item for each round specifying the minimum number of
        epochs to train for (see `Trainer.train_for(min_epochs)`)
    :param improve_epochs: A list of integers with an item for each round specifying the number of
        epochs that should surpass without validation score improvement for early termination of
        training (see `Trainer.train_for(val_improve_num_epochs)`)
    :param validation_intervals: A list of integers with an item for each round specifying the
        frequency with which validation error should be computed
        (see `Trainer.train_for(validation_interval)`)
    :param batch_xform_fn: [optional] batch transformation function to transform a batch
        before using it
    :param n_train_repetitions_in_case_of_failure: if training fails with a `TrainingFailedException`,
        this is the number of times training will be re-attempted
    :return: tuple `(classifier, indices_labelled_history, validation_error_history, test_error_history)`
        where `classifier` is the classifier trained in the final active learning iteration,
        `indices_labelled_history` is a list of numpy arrays - one for each active learning iteration - where each
        array gives the indices of the samples that were labelled in that iteration,
        `validation_error_history` is a list of validation errors, where each
    """
    assert len(num_epochs) == len(subset_sizes)
    assert len(min_epochs) == len(subset_sizes)
    assert len(improve_epochs) == len(subset_sizes)
    assert len(validation_intervals) == len(subset_sizes)

    # Classifier
    clf = None

    # Error history for reporting, along with the history of the indices of samples that were labelled
    validation_error_history = []
    test_error_history = []
    indices_labelled_history = []

    best_val_loss = best_val_err = None

    # The indices of as-of-yet unused samples
    indices_unlabelled = None

    # Training dataset
    train_ds = None

    # A mask array consisting of booleans that indicate if a sample has been
    # chosen yet
    tr_mask = np.zeros((N_train,), dtype=bool)

    # The number of samples labelled in the previous round
    N_labelled_prev_round = 0
    for round, (N_labelled, N_epochs, min_N_epochs, N_improve_epochs, val_interval) in \
            enumerate(zip(subset_sizes, num_epochs, min_epochs, improve_epochs, validation_intervals)):
        # Select samples
        if N_labelled_prev_round == 0:
            # Mark the first `N_subset` samples as chosen
            tr_mask[:N_labelled] = True
        else:
            assert train_ds is not None
            # Determine the number of additional samples being labelled
            N_additional = N_labelled - N_labelled_prev_round

            # Get the unused training samples
            train_ds_unlabelled = datasets_fn(indices_unlabelled)[0]
            N_unlabelled = indices_unlabelled.shape[0]

            # Use the sample chooser to determine which additional samples to choose
            chosen_unlabelled_sample_indices = sample_chooser.choose_samples(N_additional,
                    train_ds_unlabelled, N_unlabelled, clf, batch_xform_fn)
            print('Round {0}: N_unused={1}, chosen_unused_sample_indices.shape={2}, '
                  'chosen_unused_sample_indices.max={3}'.format(round, N_unlabelled,
                        chosen_unlabelled_sample_indices.shape, chosen_unlabelled_sample_indices.max()
            ))

            # The indices in `chosen_unused_sample_indices` are indices into `train_ds_unused`
            # which in turn are chosen using the indices from `indices_unused`, so
            # map them back to the complete training set
            chosen_sample_indices = indices_unlabelled[chosen_unlabelled_sample_indices]
            # Mark the additional samples as chosen
            tr_mask[chosen_sample_indices] = True

        print('Round {0}: {1} samples; added {2}'.format(round, N_labelled, N_labelled - N_labelled_prev_round))

        N_labelled_prev_round = N_labelled

        # Get the training set for this round by selecting masked samples
        indices_labelled = np.arange(N_train)[tr_mask]
        indices_unlabelled = np.arange(N_train)[~tr_mask]

        # Build train set for the given indices (validation and test sets always remain the same)
        train_ds, val_ds, test_ds = datasets_fn(indices_labelled)

        if refine and clf is not None:
            # Refine an existing network
            print('Refining with {0} samples for min {1} epochs max {2} epochs, terminating if no improvement after '
                  '{3} epochs validating every {4}...'.format(train_ds.num_examples, min_N_epochs, N_epochs,
                                                              N_improve_epochs, val_interval))

            # Set training length
            clf.trainer.train_for(num_epochs=N_epochs, min_epochs=min_N_epochs,
                                  val_improve_num_epochs=N_improve_epochs, validation_interval=val_interval)

            # Train
            try:
                res = clf.trainer.train(train_ds, val_ds, test_ds, batchsize=batchsize)
            except trainer.TrainingFailedException as e:
                # If `TrainingFailedException` is raised; just do nothing; its unlikely that trying again will
                # change anything
                pass
        else:
            # Training repetition loop: start training over from scratch until it succeeds, which it should most
            # of the time
            for train_rep in range(n_train_repetitions_in_case_of_failure):
                # Build the image classifier for the given model builder
                clf = image_classifier.ImageClassifier.for_model(model_builder)

                # Set verbosity
                clf.trainer.report(verbosity=trainer.VERBOSITY_MINIMAL)

                # Set data transformation function
                clf.trainer.data_xform_fn(batch_xform_fn=batch_xform_fn)

                print('Training with {0} samples for min {1} epochs max {2} epochs, terminating if no improvement after '
                      '{3} epochs validating every {4}...'.format(train_ds.num_examples, min_N_epochs, N_epochs,
                                                                  N_improve_epochs, val_interval))

                # Set training length
                clf.trainer.train_for(num_epochs=N_epochs, min_epochs=min_N_epochs,
                                      val_improve_num_epochs=N_improve_epochs, validation_interval=val_interval)

                # Train
                try:
                    res = clf.trainer.train(train_ds, val_ds, test_ds, batchsize=batchsize)
                except trainer.TrainingFailedException as e:
                    # Training failed: let the training repetition loop try again
                    pass
                else:
                    # Training succeeded: break out of the training repetition loop
                    break

        print ''

        validation_error_history.append(res.best_validation_results[1])
        test_error_history.append(res.final_test_results[1])
        indices_labelled_history.append(indices_labelled.copy())

    return clf, indices_labelled_history, validation_error_history, test_error_history

