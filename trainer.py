import sys, time, re
import numpy as np
import lasagne
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.datasets import Dataset


VERBOSITY_NONE = None
VERBOSITY_MINIMAL = 'minimal'
VERBOSITY_EPOCH = 'epoch'
VERBOSITY_BATCH = 'batch'


def _is_sequence_of_arrays(dataset):
    if isinstance(dataset, (tuple, list)):
        for x in dataset:
            if not isinstance(x, np.ndarray):
                return False
        return True
    return False


def iterate_minibatches(data, batchsize, shuffle=False):
    N = data[0].shape[0]
    for d1 in data[1:]:
        assert d1.shape[0] == N
    if shuffle:
        indices = np.arange(N)
        np.random.shuffle(indices)
    for start_idx in range(0, N - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [d[excerpt] for d in data]


def _default_epoch_log_fn(epoch_number, delta_time, train_results, val_results, test_results):
    if val_results is None:
        return 'Epoch {} ({:.2f}s): train {}'.format(epoch_number, delta_time, train_results)
    elif test_results is None:
        return 'Epoch {} ({:.2f}s): train {}, validation {}'.format(epoch_number, delta_time, train_results,
                                                                    val_results)
    else:
        return 'Epoch {} ({:.2f}s): train {}, validation {} test {}'.format(epoch_number, delta_time, train_results,
                                                                            val_results, test_results)


class TrainingFailedException (Exception):
    """
    This exception is raised to indicate that training failed for some reason; often indicating that the training

    :var epoch: the epoch at which training failed
    :var reason: a string providing the reason that training failed
    :var parameters_reset: a boolean indicating if the network parameters were successfully reset to the initial state
    """
    def __init__(self, epoch, reason, parameters_reset):
        super(TrainingFailedException, self).__init__('Training failed at epoch {}: {}'.format(epoch, reason))
        self.epoch = epoch
        self.reason = reason
        self.parameters_reset = parameters_reset


class TrainingResults (object):
    """
    `TrainingResults` instance provide the results of training a neural network using a `Trainer` instance.

    :var train_results: list giving the per epoch results of the training function
    :var validation_results: list giving the per epoch results of the evaluation function applied to the validation set
    """
    def __init__(self, train_results, validation_results, best_val_epoch, best_validation_results, test_results,
                 final_test_results, last_epoch):
        self.train_results = train_results
        self.validation_results = validation_results
        self.best_val_epoch = best_val_epoch
        self.best_validation_results = best_validation_results
        self.test_results = test_results
        self.final_test_results = final_test_results
        self.last_epoch = last_epoch


class Trainer (object):
    def __init__(self):
        """
        Constructor for a neural network trainer. The `Trainer` class provides training loop functionaltiy.
        It is written to be as generic as possible in order to simplify implementing a Theano/Lasagne
        training loop.

        You must at least invoke the `train_with` method to set the training function.
        """
        self.fuel_stream_xform_fn = None
        self.batch_xform_fn = None

        self.train_batch_fn = None
        self.train_epoch_results_check_fn = None

        self.eval_batch_fn = None
        self.validation_score_fn = lambda x: x[0]
        self.validation_interval = None

        self.num_epochs = 200
        self.min_epochs = None
        self.val_improve_num_epochs = 0
        self.val_improve_epochs_factor = 0

        self.verbosity = VERBOSITY_EPOCH
        self.epoch_log_fn = None
        self.log_stream = sys.stdout

        self.get_state_fn = None
        self.set_state_fn = None


    def train_with(self, train_batch_fn, train_epoch_results_check_fn=None):
        """
        Set the batch training function. This method *MUST* be called before attempting to use the
        trainer.

        :param train_batch_fn: mini-batch training function that updates the network parameters,
        of the form `f(*batch_data) -> results` that where `batch_data` is a list of numpy arrays
        that contain the training data for the batch and `results` is a list of floats/numpy arrays that
        represent loss/error rates/etc, or `None`. Note that the training function results should
        represent the *sum* of the loss/error rate for that batch as the values will be accumulated
        and divided by the total number of training samples after all mini-batches has been processed.
        :param train_epoch_results_check_fn: [optional] a function of the form
        `f(epoch, train_epoch_results) -> error_reason` that is invoked to check the results returned by
        `train_batch_fn` accumulated during the epoch; if no training failure is detected, it should
        return `None`, whereas if a failure is detected - e.g. training loss is NaN - it should return
        a reason string that will be used to build a `TrainingFailedException` that will be raised by
        the `train` method. Note that should a training failure be detected, the trainer will attempt
        to restore the network's parameters to the same values it had before the `train` method
        was invoked.
        :return: `self`
        """
        self.train_batch_fn = train_batch_fn
        self.train_epoch_results_check_fn = train_epoch_results_check_fn
        return self

    def evaluate_with(self, eval_batch_fn, validation_score_fn=0):
        """
        Set the batch validation/test function.

        :param eval_batch_fn: mini-batch evaluation function (for validation/test) of the form
        `f(*batch_data) -> results` where `batch_data` is a list of numpy arrays that contain the
        data to evaluate for the batch and `results` is a list of floats that represent loss/error
        rates/etc, or `None`. Note that the evaluation function results should represent the *sum*
        of the loss/error rate for that batch as the values will be accumulated and divided by the
        total number of training samples at the end. Note that for the purpose of detecting improvements
        in validation, *better* results should have *lower* values, or use the `validation_score_fn`
        to provide a function that negates/inverts the score.
        :param validation_score_fn: Either an integer index or a callable; if an index then the
        score is obtained via `validation_results[validation_score_fn]`, if a callable then the
        score is obtained via `validation_score_fn(validation_results)`.
        :return: `self`
        """
        self.eval_batch_fn = eval_batch_fn
        if isinstance(validation_score_fn, (int, long)):
            self.validation_score_fn = lambda x: x[validation_score_fn]
        elif callable(validation_score_fn):
            self.validation_score_fn = validation_score_fn
        else:
            raise TypeError('validation_score should be an integer index or a function')
        return self

    def train_for(self, num_epochs, min_epochs=None, val_improve_num_epochs=0, val_improve_epochs_factor=0,
                  validation_interval=None):
        """
        Set the number of epochs for which training should proceed and set early termination conditions.

        Training will proceed for at most `num_epochs` epochs.
        If the validation score improves at epoch `p`, then training will terminate at epoch
        `max(p + val_improve_num_epochs, p * val_improve_epochs_factor, min_epochs)`
        if no further improvement in validation score is detected in the mean time.

        :param num_epochs: The maximum number of epochs to train for. Training will run for this
        many epochs unless early termination is set up using the `early_termination` method.
        :param min_epochs: the minimum number of epochs to train for
        :param val_improve_num_epochs: a fixed number of epochs after which training should terminate
        if not validation score improvement is detected
        :param val_improve_epochs_factor: a factor by which the training time increases when
        a validation score improvement is detected
        :param validation_interval: Perform validation every `validation_interval` epochs, or `None`
        for every epoch
        :return: `self`
        """
        if min_epochs is None:
            self.min_epochs = num_epochs
        else:
            self.min_epochs = min(min_epochs, num_epochs)
        self.num_epochs = num_epochs
        self.val_improve_num_epochs = val_improve_num_epochs
        self.val_improve_epochs_factor = val_improve_epochs_factor
        self.validation_interval = validation_interval
        return self

    def report(self, verbosity=VERBOSITY_EPOCH, epoch_log_fn=None, log_stream=None):
        """
        Set up reporting and verbosity.

        Logs output to `sys.stdout` by default, unless an alte

        :param verbosity: how much information is written to the logging stream describing progress.
        If `VERBOSITY_NONE` then nothing is reported. If `VERBOSITY_MINIMAL` then a `.` is written
        per epoch, or a `*` when the validation score improves. If `VERBOSITY_EPOCH` then
        a single line report is produced for each epoch. If `VERBOSITY_BATCH` then in addition
        to a single line per epoch
        :param epoch_log_fn: a function that creates a log entry to display an epoch result of the form
        `fn(epoch_number, delta_time, train_results, val_results, test_results) -> str` that
        generates a string describing the epoch training results
        :param log_stream: a file-like object to which progress is to be written.
        :return: `self`
        """
        self.verbosity = verbosity
        self.epoch_log_fn = epoch_log_fn or self.epoch_log_fn
        self.log_stream = log_stream or self.log_stream
        return self

    def retain_best_scoring_state_of_updates(self, updates):
        """
        Provide an updates dict or tuple list that is passed as the `updates` parameter to
        `theano.function` and often generated by functions from `lasagne.updates` that is
        used to acquire parameters whose values are saved and restored in order to
        save and restore the network; see the `keep_best_scoring_state` method
        :param layer: a `lasagne.layers.Layer`.
        """
        if isinstance(updates, dict):
            params = list(updates.keys())
        elif isinstance(updates, (list, tuple)):
            params = [upd[0] for upd in updates]
        else:
            raise TypeError('updates should be a dict mapping parameter to update expression '
                            'or a sequence of tuples of parameter, update expression pairs')

        def get_state():
            return [p.get_value() for p in params]

        def set_state(state):
            for p, v in zip(params, state):
                p.set_value(v)

        self.retain_best_scoring_state(get_state, set_state)
        return self

    def retain_best_scoring_state_of_network(self, layer):
        """
        Provide a lasagne layer (`lasagne.layers.Layer`) that is used to saving and restoring the
        state of the network; see the `keep_best_scoring_state` method

        NOTE: the `retain_best_scoring_state_of_updates` method is preferred since the state
        of additional variables created by the update function are also saved and restored by
        this method.

        :param layer: a `lasagne.layers.Layer`.
        """
        get_state = lambda: lasagne.layers.get_all_param_values(layer, trainable=True)
        set_state = lambda state: lasagne.layers.set_all_param_values(layer, state, trainable=True)
        self.retain_best_scoring_state(get_state, set_state)
        return self

    def retain_best_scoring_state(self, get_state_fn, set_state_fn):
        """
        Provide functions for getting and setting the state (weights, biases, etc) of the network
        currently being trained, so that the trainer can keep the state of the network at the epoch
        that gave the best validation score
        :param get_state_fn: a function of the form `fn() -> state` that gets the state of the network
        :param set_state_fn: a function of the form `fn(state)` that sets the state of the network
        """
        self.get_state_fn = get_state_fn
        self.set_state_fn = set_state_fn
        return self

    def data_xform_fn(self, fuel_stream_xform_fn=None, batch_xform_fn=None):
        """
        Set data transformation functions; can provide a Fuel `DataStream` transformation function
        and a batch transformation function.

        :param fuel_stream_xform_fn: a function of the form `fn(stream) -> transformed_stream`
        :param set_state_fn: a function of the form `fn(batch) -> transformed_batch`
        """
        self.fuel_stream_xform_fn = fuel_stream_xform_fn
        self.batch_xform_fn = batch_xform_fn
        return self


    def train(self, train_set, val_set, test_set, batchsize):
        """
        Run the training loop.

        :param train_set: the training set
        :param val_set: the validation set, or `None`
        :param test_set: the test set, or `None`
        :param batchsize: the mini-batch size
        :return: a `TrainingResults` instance that provides the per-epoch history of results of
        training, validation and testing, along with the epoch that gave the best validation score
        and the best validation and final test results.
        """
        if self.train_batch_fn is None:
            raise ValueError('no batch training function provided; call the `train_with` method '
                             'before starting the training loop')
        if val_set is not None and self.eval_batch_fn is None:
            raise ValueError('validation set provided but no evaluation function available')
        if test_set is not None and self.eval_batch_fn is None:
            raise ValueError('test set provided but no evaluation function available')

        stop_at_epoch = self.min_epochs
        epoch = 0

        # If we have a training results check function, save the state
        if self.train_epoch_results_check_fn is not None:
            state_at_start = self._save_state()
        else:
            state_at_start = None

        best_validation_results = None
        best_epoch = None
        best_state = None
        state_saved = False
        test_results = None

        all_train_results = []
        all_val_results = [] if val_set is not None and self.eval_batch_fn is not None else None
        all_test_results = [] if test_set is not None and self.eval_batch_fn is not None else None

        train_start_time = time.time()

        while epoch < min(stop_at_epoch, self.num_epochs):
            epoch_start_time = time.time()

            # TRAIN
            # Log start of training
            if self.verbosity == VERBOSITY_BATCH:
                self._log('[')
                on_train_batch = lambda: self._log_train_batch(epoch)
            else:
                on_train_batch = None

            # Train
            train_results = self._batch_loop(self.train_batch_fn, train_set, batchsize, shuffle=True,
                                             on_complete_batch=on_train_batch)

            if self.train_epoch_results_check_fn is not None:
                reason = self.train_epoch_results_check_fn(epoch, train_results)
                if reason is not None:
                    # Training failed: attempt to restore parameters to initial state
                    if state_at_start is not None:
                        params_restored = self._restore_state(state_at_start)
                    else:
                        params_restored = False

                    if self.verbosity != VERBOSITY_NONE:
                        self._log("Training failed at epoch{}: {}\n".format(epoch, reason))

                    raise TrainingFailedException(epoch, reason, params_restored)

            # Log the end of training
            if self.verbosity == VERBOSITY_BATCH:
                self._log(']\n')


            validated = False
            tested = False
            validation_improved = False
            # VALIDATION
            if val_set is not None and self._should_validate(epoch):
                validated = True
                validation_results = self._batch_loop(self.eval_batch_fn, val_set, batchsize,
                                                      shuffle=False)

                if best_validation_results is None or \
                        self.validation_score_fn(validation_results) <\
                                self.validation_score_fn(best_validation_results):
                    validation_improved = True

                    # Validation score improved
                    best_validation_results = validation_results
                    best_epoch = epoch
                    best_state = self._save_state()
                    state_saved = True

                    stop_at_epoch = max(epoch + 1 + self.val_improve_num_epochs,
                                        (epoch + 1) * self.val_improve_epochs_factor,
                                        self.min_epochs)

                    if test_set is not None:
                        tested = True
                        test_results = self._batch_loop(self.eval_batch_fn, test_set, batchsize,
                                                        shuffle=False)
            else:
                validation_results = None

            if not tested and test_set is not None and val_set is None:
                tested = True
                test_results = self._batch_loop(self.eval_batch_fn, test_set, batchsize,
                                                shuffle=False)

            if self.verbosity == VERBOSITY_BATCH or self.verbosity == VERBOSITY_EPOCH:
                self._log_epoch_results(epoch, time.time() - epoch_start_time, train_results,
                                        validation_results if validated else None,
                                        test_results if tested else None)
            elif self.verbosity == VERBOSITY_MINIMAL:
                if validation_improved:
                    self._log('*')
                elif validated:
                    self._log('-')
                else:
                    self._log('.')

            all_train_results.append(train_results)
            if all_val_results is not None:
                all_val_results.append(validation_results)
            if all_test_results is not None:
                all_test_results.append(test_results)
            epoch += 1

        train_end_time = time.time()

        if state_saved:
            self._restore_state(best_state)

        if self.verbosity != VERBOSITY_BATCH and self.verbosity != VERBOSITY_EPOCH:
            final_train_results = all_train_results[-1] if len(all_train_results) > 0 else None
            self._log("Final result:\n")
            self._log_epoch_results(epoch, train_end_time - train_start_time, final_train_results,
                                    best_validation_results, test_results)


        return TrainingResults(
            train_results=all_train_results,
            validation_results=all_val_results,
            best_validation_results=best_validation_results,
            best_val_epoch=best_epoch,
            test_results=all_test_results,
            final_test_results=test_results,
            last_epoch=epoch,
        )


    def _log(self, text):
        log_stream = self.log_stream or sys.stdout
        log_stream.write(text)
        log_stream.flush()

    def _log_train_batch(self, epoch):
        if self.verbosity == VERBOSITY_BATCH:
            self._log('.')

    def _should_validate(self, epoch):
        return self.validation_interval is None  or  epoch % self.validation_interval == 0

    def _log_epoch_results(self, epoch, delta_time, train_results, val_results, test_results):
        epoch_log_fn = self.epoch_log_fn or _default_epoch_log_fn
        self._log(epoch_log_fn(epoch, delta_time, train_results, val_results, test_results) + '\n')

    def _save_state(self):
        if self.get_state_fn is not None:
            return self.get_state_fn()
        else:
            return None

    def _restore_state(self, state):
        if self.set_state_fn is not None:
            self.set_state_fn(state)
            return True
        else:
            return False


    def batch_iterator(self, dataset, batchsize, shuffle=False):
        if isinstance(dataset, Dataset):
            if shuffle:
                train_scheme = ShuffledScheme(examples=dataset.num_examples, batch_size=batchsize)
            else:
                train_scheme = SequentialScheme(examples=dataset.num_examples, batch_size=batchsize)
            # Use `DataStream.default_stream`, otherwise the default transformers defined by the dataset *wont*
            # be applied
            stream = DataStream.default_stream(dataset=dataset, iteration_scheme=train_scheme)
            if self.fuel_stream_xform_fn is not None:
                stream = self.fuel_stream_xform_fn(stream)
            return stream.get_epoch_iterator()
        elif _is_sequence_of_arrays(dataset):
            return iterate_minibatches(dataset, batchsize, shuffle=shuffle)
        else:
            raise TypeError('dataset should be a fuel Dataset instance or a list of arrays')


    def _batch_loop(self, fn, data, batchsize, shuffle=False, on_complete_batch=None):
        # Accumulator for results and number of samples
        results_accum = None
        n_samples_accum = 0

        # Train on each batch
        for batch in self.batch_iterator(data, batchsize, shuffle=shuffle):
            # Aply batch transformation function
            if self.batch_xform_fn is not None:
                batch = self.batch_xform_fn(batch)

            # Get number of samples in batch; can vary
            batch_N = batch[0].shape[0]

            # Apply on batch and check the type of the results
            batch_results = fn(*batch)
            if batch_results is None:
                pass
            elif isinstance(batch_results, np.ndarray):
                batch_results = [batch_results]
            elif isinstance(batch_results, list):
                pass
            else:
                raise TypeError('Batch function should return a list of results for the batch or None, not {}'.format(type(batch_results)))

            # Accumulate training results and number of examples
            if results_accum is None:
                results_accum = batch_results
            else:
                if batch_results is not None:
                    for i in range(len(results_accum)):
                        results_accum[i] += batch_results[i]
            n_samples_accum += batch_N

            if on_complete_batch is not None:
                on_complete_batch()

        # Divide by the number of training examples used to compute mean
        if results_accum is not None:
            results_accum = [r / n_samples_accum for r in results_accum]

        return results_accum


import unittest, cStringIO

class Test_Trainer (unittest.TestCase):
    class TrainFunction (object):
        def __init__(self, results=None, states=None, on_invoke=None, result_repeat=1):
            self.count = 0
            self.results = results
            self.states = states
            self.current_state = None
            self.state_get_count = 0
            self.state_set_count = 0
            self.on_invoke = on_invoke
            self.result_repeat = result_repeat

        def get_state(self):
            self.state_get_count += 1
            return self.current_state

        def set_state(self, state):
            self.state_set_count += 1
            self.current_state = state

        def __call__(self, *args, **kwargs):
            d0 = args[0]
            batch_N = d0.shape[0]
            i = self.count
            if self.on_invoke is not None:
                self.on_invoke()
            self.count += 1
            if self.states is not None:
                self.current_state = self.states[i]
            if self.results is None:
                return None
            else:
                return [r*batch_N for r in self.results[i/self.result_repeat]]



    def test_no_train_fn(self):
        trainer = Trainer()

        self.assertRaises(ValueError, lambda: trainer.train([np.arange(10)], [np.arange(10)], None, 5))


    def test_no_eval_fn(self):
        log = cStringIO.StringIO()
        train_fn = self.TrainFunction()

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=200)
        trainer.report(log_stream=log, verbosity=VERBOSITY_NONE)

        self.assertRaises(ValueError, lambda: trainer.train([np.arange(10)], [np.arange(10)], None, 5))
        self.assertRaises(ValueError, lambda: trainer.train([np.arange(10)], None, [np.arange(10)], 5))


    def test_train(self):
        log = cStringIO.StringIO()
        train_fn = self.TrainFunction()

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=200)
        trainer.retain_best_scoring_state(train_fn.get_state, train_fn.set_state)
        trainer.report(log_stream=log, verbosity=VERBOSITY_NONE)

        res = trainer.train([np.arange(10)], None, None, 5)

        # Called 400 times - 2x per epoch for 200 epochs
        self.assertEqual(train_fn.count, 400)
        self.assertEqual(log.getvalue(), '')
        self.assertEqual(train_fn.state_get_count, 0)
        self.assertEqual(train_fn.state_set_count, 0)
        self.assertEqual(len(res.train_results), 200)
        self.assertEqual(res.validation_results, None)
        self.assertEqual(res.test_results, None)
        self.assertEqual(res.best_val_epoch, None)
        self.assertEqual(res.best_validation_results, None)
        self.assertEqual(res.final_test_results, None)
        self.assertEqual(res.last_epoch, 200)



    def test_validate_with(self):
        log = cStringIO.StringIO()
        val_out = [[i] for i in xrange(200)]
        train_fn = self.TrainFunction()
        eval_fn = self.TrainFunction(val_out)

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=200)
        trainer.evaluate_with(eval_fn)
        trainer.report(log_stream=log, verbosity=VERBOSITY_NONE)

        trainer.train([np.arange(10)], [np.arange(5)], None, 5)

        # Called 400 times - 2x per epoch for 200 epochs
        self.assertEqual(train_fn.count, 400)
        self.assertEqual(eval_fn.count, 200)
        self.assertEqual(log.getvalue(), '')
        self.assertEqual(train_fn.state_get_count, 0)
        self.assertEqual(train_fn.state_set_count, 0)


    def test_validate_with_store_state(self):
        log = cStringIO.StringIO()
        train_fn = self.TrainFunction()
        eval_fn = self.TrainFunction([[i] for i in xrange(200)])

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=200)
        trainer.evaluate_with(eval_fn)
        trainer.retain_best_scoring_state(train_fn.get_state, train_fn.set_state)
        trainer.report(log_stream=log, verbosity=VERBOSITY_NONE)

        trainer.train([np.arange(10)], [np.arange(5)], None, 5)

        # Called 400 times - 2x per epoch for 200 epochs
        self.assertEqual(train_fn.count, 400)
        self.assertEqual(eval_fn.count, 200)
        self.assertEqual(log.getvalue(), '')
        self.assertEqual(train_fn.state_get_count, 1)
        self.assertEqual(train_fn.state_set_count, 1)


    def test_validation_interval(self):
        log = cStringIO.StringIO()
        train_fn = self.TrainFunction()
        def on_eval():
            self.assertTrue(train_fn.count in range(1, 201, 10))
        eval_fn = self.TrainFunction([[i] for i in xrange(200)], on_invoke=on_eval)

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=200, validation_interval=10)
        trainer.evaluate_with(eval_fn)
        trainer.report(log_stream=log, verbosity=VERBOSITY_NONE)

        trainer.train([np.arange(5)], [np.arange(5)], None, 5)

        self.assertEqual(train_fn.count, 200)
        self.assertEqual(eval_fn.count, 20)
        self.assertEqual(log.getvalue(), '')


    def test_validation_score_index(self):
        val_output = zip(range(200), range(101,1,-1) + range(1,101,1))
        val_output = [list(xs) for xs in val_output]
        log = cStringIO.StringIO()
        train_fn = self.TrainFunction()
        eval_fn = self.TrainFunction(val_output)

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=200)
        trainer.evaluate_with(eval_fn, validation_score_fn=1)
        trainer.report(log_stream=log, verbosity=VERBOSITY_NONE)

        res = trainer.train([np.arange(5)], [np.arange(5)], None, 5)

        self.assertEqual(train_fn.count, 200)
        self.assertEqual(eval_fn.count, 200)
        self.assertEqual(log.getvalue(), '')

        self.assertEqual(res.validation_results, val_output)
        self.assertEqual(res.best_validation_results, [100,1])


    def test_validation_score_fn(self):
        val_output = zip(range(200), range(101,1,-1) + range(1,101,1))
        val_output = [list(xs) for xs in val_output]
        log = cStringIO.StringIO()
        train_fn = self.TrainFunction()
        eval_fn = self.TrainFunction(val_output)

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=200)
        trainer.evaluate_with(eval_fn, validation_score_fn=lambda val_res: val_res[1])
        trainer.report(log_stream=log, verbosity=VERBOSITY_NONE)

        res = trainer.train([np.arange(5)], [np.arange(5)], None, 5)

        self.assertEqual(train_fn.count, 200)
        self.assertEqual(eval_fn.count, 200)
        self.assertEqual(log.getvalue(), '')

        self.assertEqual(res.validation_results, val_output)
        self.assertEqual(res.best_validation_results, [100,1])


    def test_train_for_num_epochs(self):
        val_output = zip(range(200), range(101,1,-1) + range(1,101,1))
        val_output = [list(xs) for xs in val_output]
        log = cStringIO.StringIO()
        train_fn = self.TrainFunction()
        eval_fn = self.TrainFunction(val_output)

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=150)
        trainer.evaluate_with(eval_fn, validation_score_fn=lambda val_res: val_res[1])
        trainer.report(log_stream=log, verbosity=VERBOSITY_NONE)

        res = trainer.train([np.arange(5)], [np.arange(5)], None, 5)

        self.assertEqual(train_fn.count, 150)
        self.assertEqual(eval_fn.count, 150)
        self.assertEqual(log.getvalue(), '')

        self.assertEqual(res.validation_results, val_output[:150])
        self.assertEqual(res.best_validation_results, [100,1])


    def test_train_for_min_epochs(self):
        val_output = zip(range(200), range(101,1,-1) + range(1,101,1))
        val_output = [list(xs) for xs in val_output]
        log = cStringIO.StringIO()
        train_fn = self.TrainFunction()
        eval_fn = self.TrainFunction(val_output)

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=200, min_epochs=95)
        trainer.evaluate_with(eval_fn, validation_score_fn=lambda val_res: val_res[1])
        trainer.report(log_stream=log, verbosity=VERBOSITY_NONE)

        res = trainer.train([np.arange(5)], [np.arange(5)], None, 5)

        self.assertEqual(train_fn.count, 95)
        self.assertEqual(eval_fn.count, 95)
        self.assertEqual(log.getvalue(), '')

        self.assertEqual(res.validation_results, val_output[:95])
        self.assertEqual(res.best_validation_results, [94,7])


    def test_train_for_val_improve_num_epochs(self):
        val_output = zip(range(200), range(101,1,-1) + range(1,101,1))
        val_output = [list(xs) for xs in val_output]
        log = cStringIO.StringIO()
        train_fn = self.TrainFunction()
        eval_fn = self.TrainFunction(val_output)

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=200, min_epochs=95, val_improve_num_epochs=10)
        trainer.evaluate_with(eval_fn, validation_score_fn=lambda val_res: val_res[1])
        trainer.report(log_stream=log, verbosity=VERBOSITY_NONE)

        res = trainer.train([np.arange(5)], [np.arange(5)], None, 5)

        self.assertEqual(train_fn.count, 111)
        self.assertEqual(eval_fn.count, 111)
        self.assertEqual(log.getvalue(), '')

        self.assertEqual(res.validation_results, val_output[:111])
        self.assertEqual(res.best_validation_results, [100,1])


    def test_train_for_val_improve_epochs_factor(self):
        val_output = zip(range(200), range(75,0,-1) + range(0,125,1))
        val_output = [list(xs) for xs in val_output]
        log = cStringIO.StringIO()
        train_fn = self.TrainFunction()
        eval_fn = self.TrainFunction(val_output)

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=200, min_epochs=65, val_improve_epochs_factor=2)
        trainer.evaluate_with(eval_fn, validation_score_fn=lambda val_res: val_res[1])
        trainer.report(log_stream=log, verbosity=VERBOSITY_NONE)

        res = trainer.train([np.arange(5)], [np.arange(5)], None, 5)

        self.assertEqual(train_fn.count, 152)
        self.assertEqual(eval_fn.count, 152)
        self.assertEqual(log.getvalue(), '')

        self.assertEqual(res.validation_results, val_output[:152])
        self.assertEqual(res.best_validation_results, [75,0])


    def test_report_verbosity_none(self):
        val_output = zip(range(200), range(75,0,-1) + range(0,125,1))
        val_output = [list(xs) for xs in val_output]
        log = cStringIO.StringIO()
        train_fn = self.TrainFunction()
        eval_fn = self.TrainFunction(val_output)

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=200, min_epochs=65, val_improve_epochs_factor=2)
        trainer.evaluate_with(eval_fn, validation_score_fn=lambda val_res: val_res[1])
        trainer.report(log_stream=log, verbosity=VERBOSITY_NONE)

        res = trainer.train([np.arange(5)], [np.arange(5)], None, 5)

        self.assertEqual(log.getvalue(), '')


    def test_report_verbosity_minimal(self):
        val_output = zip(range(200), range(75,0,-1) + range(0,125,1))
        val_output = [list(xs) for xs in val_output]
        log = cStringIO.StringIO()
        train_fn = self.TrainFunction()
        eval_fn = self.TrainFunction(val_output)

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=200, min_epochs=65, val_improve_epochs_factor=2)
        trainer.evaluate_with(eval_fn, validation_score_fn=lambda val_res: val_res[1])
        trainer.report(log_stream=log, verbosity=VERBOSITY_MINIMAL)

        res = trainer.train([np.arange(5)], [np.arange(5)], None, 5)

        self.assertEqual(train_fn.count, 152)
        self.assertEqual(eval_fn.count, 152)
        self.assertEqual(log.getvalue(), '*' * 76 + '.' * 76)


    def test_report_verbosity_epoch(self):
        val_output = zip(range(200), range(75,0,-1) + range(0,125,1))
        val_output = [list(xs) for xs in val_output]
        log = cStringIO.StringIO()
        train_fn = self.TrainFunction(val_output)
        eval_fn = self.TrainFunction(val_output)

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=200, min_epochs=65, val_improve_epochs_factor=2)
        trainer.evaluate_with(eval_fn, validation_score_fn=lambda val_res: val_res[1])
        trainer.report(log_stream=log, verbosity=VERBOSITY_EPOCH)

        res = trainer.train([np.arange(5)], [np.arange(5)], None, 5)

        self.assertEqual(train_fn.count, 152)
        self.assertEqual(eval_fn.count, 152)
        log_lines = log.getvalue().split('\n')
        for i, line in enumerate(log_lines):
            if line.strip() != '':
                pattern = re.escape('Epoch {0} ('.format(i)) + \
                          r'[0-9]+\.[0-9]+s' + \
                          re.escape('): train [{0}, {1}], validation [{0}, {1}]'.format(val_output[i][0], val_output[i][1]))
                match = re.match(pattern, line)
                if match is None or match.end(0) != len(line):
                    self.fail(msg=line)


    def test_report_verbosity_batch(self):
        val_output = zip(range(200), range(75,0,-1) + range(0,125,1))
        val_output = [list(xs) for xs in val_output]
        log = cStringIO.StringIO()
        train_fn = self.TrainFunction()
        eval_fn = self.TrainFunction(val_output, result_repeat=4)

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=200, min_epochs=65, val_improve_epochs_factor=2)
        trainer.evaluate_with(eval_fn, validation_score_fn=lambda val_res: val_res[1])
        trainer.report(log_stream=log, verbosity=VERBOSITY_BATCH)

        res = trainer.train([np.arange(20)], [np.arange(20)], None, 5)

        self.assertEqual(train_fn.count, 608)
        self.assertEqual(eval_fn.count, 608)
        log_lines = log.getvalue().split('\n')
        for i, (line_a, line_b) in enumerate(zip(log_lines[::2], log_lines[1::2])):
            if line_a.strip() != '' and line_b.strip() != '':
                pattern_b = re.escape('Epoch {0} ('.format(i)) + \
                          r'[0-9]+\.[0-9]+s' + \
                          re.escape('): train None, validation [{0}, {1}]'.format(val_output[i][0], val_output[i][1]))
                self.assertEqual(line_a, '[....]')
                match = re.match(pattern_b, line_b)
                if match is None or match.end(0) != len(line_b):
                    self.fail(msg=line_b)


    def test_report_epoch_log_fn(self):
        train_output = [[x] for x in range(200)]
        val_output = zip(range(200), range(75,0,-1) + range(0,125,1))
        val_output = [list(xs) for xs in val_output]
        log = cStringIO.StringIO()
        train_fn = self.TrainFunction(train_output)
        eval_fn = self.TrainFunction(val_output)

        def epoch_log_fn(epoch_number, delta_time, train_results, val_results, test_results):
            return '{}: train: {}, val: {} {}'.format(epoch_number, train_results[0], val_results[0], val_results[1])

        trainer = Trainer()
        trainer.train_with(train_batch_fn=train_fn)
        trainer.train_for(num_epochs=200, min_epochs=65, val_improve_epochs_factor=2)
        trainer.evaluate_with(eval_fn, validation_score_fn=lambda val_res: val_res[1])
        trainer.report(log_stream=log, verbosity=VERBOSITY_EPOCH, epoch_log_fn=epoch_log_fn)

        trainer.train([np.arange(5)], [np.arange(5)], None, 5)

        self.assertEqual(train_fn.count, 152)
        self.assertEqual(eval_fn.count, 152)
        log_lines = log.getvalue().split('\n')
        for i, line in enumerate(log_lines):
            if line.strip() != '':
                self.assertEqual(line, '{}: train: {}, val: {} {}'.format(i, train_output[i][0], val_output[i][0], val_output[i][1]))

