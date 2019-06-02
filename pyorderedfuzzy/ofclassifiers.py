import numpy as np
import ofrandom as ofr
from ofnumber import OFNumber, fpower, fexp
from ofmodels import OFSeries


class OFPercepton(object):
    def __init__(self, activation='heaviside'):
        super(OFPercepton, self).__init__()
        self.activation = activation
        self.weights = None

    def fit(self, data_x, targets, eta=0.01, epoch=100, verbose=1):
        nsample, n = data_x.shape
        dim = data_x[0, 0].branch_f.dim

        # initial weights
        weights = ofr.ofnormal_sample(n, OFNumber.init_from_scalar(0.0, dim=dim),
                                      OFNumber.init_from_scalar(0.0, dim=dim), 1, 0.5).values

        i = 0
        err = 0
        while i < epoch:
            i += 1
            if self.activation == 'heaviside':
                for p in range(nsample):
                    y = heaviside(data_x[p], weights)
                    weights = weights + eta * (targets[p]-y) * data_x[p]
                err = 0
                for p in range(nsample):
                    y = heaviside(data_x[p], weights)
                    err += abs(targets[p]-y)
            elif self.activation == 'sigmoid':
                for p in range(nsample):
                    y = heaviside(data_x[p], weights)
                    weights = weights + eta * (targets[p]-y) * data_x[p]
                err = 0
                for p in range(nsample):
                    y = sigmoid(data_x[p], weights)
                    err += fpower(targets[p]-y, 2)
            else:
                raise ValueError('Unsupported activation method')
            if i % verbose == 0 and verbose > 0:
                print('Epoch: {}, Error: {}'.format(i, err))
            if err == 0:
                break
        if verbose >= 0:
            print('After {} epochs, Error: {}'.format(i, err))
        self.weights = OFSeries(weights)

    def predict(self, data_x):
        return np.apply_along_axis(heaviside, 1, data_x, args=self.weights)


def heaviside(x, w):
    return 1 if np.sum(w * x) >= 0 else 0


def sigmoid(x, w):
    s = np.sum(x * w)
    return 1/(1 + fexp(-s))


def sigmaid_diff(x, w):
    s = sigmoid(x, w)
    return s*(1-s)
