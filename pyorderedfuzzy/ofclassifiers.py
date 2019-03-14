import numpy as np
import ofrandom as ofr
from ofnumber import OFNumber
from ofmodels import OFSeries


class OFPercepton(object):
    def __init__(self):
        super(OFPercepton, self).__init__()

    def fit(self, data_x, targets, eta=0.01, epoch=100, vervose=1):
        nsample, n = data_x.shape
        dim = data_x[0,0].branch_f.dim

        # initial weights
        weights = ofr.ofnormal_sample(n, OFNumber.init_from_scalar(0.0, dim=dim),
                                      OFNumber.init_from_scalar(0.0, dim=dim), 1, 0.5).values

        i = 0
        err = 0
        while i < epoch:
            i += 1
            for p in range(nsample):
                y = heaviside(data_x[p], weights)
                weights = weights + eta * (targets[p]-y) * data_x[p]
            err = 0
            for p in range(nsample):
                y = heaviside(data_x[p], weights)
                err += abs(targets[p]-y)
            if i % vervose == 0 and vervose > 0:
                print('Epoach: {}, Error: {}'.format(i, err))
            if err == 0:
                break
        if vervose >= 0:
            print('Finish after {} epochs, Error: {}'.format(i, err))
        self.weights = OFSeries(weights)

    def predict(self, data_x):
        return np.apply_along_axis(heaviside, 1, data_x, args=self.weights)


def heaviside(x, w):
    return 1 if np.sum(w * x) >= 0 else 0
