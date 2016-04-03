import numpy as np

from fuel.datasets.mnist import MNIST

import utils


class MNISTTrainValTest (object):
    def __init__(self):
        self.train = MNIST(which_sets=['train'], subset=slice(0, 50000), load_in_memory=False)
        self.val = MNIST(which_sets=['train'], subset=slice(50000, None), load_in_memory=False)
        self.test = MNIST(which_sets=['test'], load_in_memory=False)

        self.train_set_indices = np.arange(50000)
        d_y = MNIST(which_sets=['train'], sources=['targets'], subset=slice(0, 50000),
                    load_in_memory=True)
        self.train_y = d_y.get_data(d_y.open(), slice(None))[0]


    def balanced_train_subset_indices(self, N_train):
        return utils.balanced_subset_indices(self.train_y, 10, N_train)


    def datasets(self, train_subset_indices=None):
        train_set_indices = self.train_set_indices
        if train_subset_indices is not None:
            train_set_indices = self.train_set_indices[train_subset_indices]

        train = MNIST(which_sets=('train',), subset=list(train_set_indices), load_in_memory=False)

        return train, self.val, self.test
