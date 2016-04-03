import numpy as np
import lasagne


def load_model(path, network):
    with np.load(path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    lasagne.layers.set_all_param_values(network, param_values)

def save_model(path, network):
    np.savez(path, *lasagne.layers.get_all_param_values(network))

def get_network_input_var(network):
    layers = lasagne.layers.get_all_layers(network)
    input_layers = [l for l in layers if isinstance(l, lasagne.layers.InputLayer)]
    if len(input_layers) == 1:
        return input_layers[0].input_var
    else:
        raise ValueError('Could not find unique input layer in network')


def balanced_subset_indices(y, n_classes, n_samples):
    n_per_class = n_samples / n_classes
    indices = np.arange(y.shape[0])
    selected_indices = []
    for cls_index in range(n_classes):
        indices_in_cls = indices[y==cls_index]
        selected_indices.append(indices_in_cls[:n_per_class])
    return np.concatenate(selected_indices, axis=0)
