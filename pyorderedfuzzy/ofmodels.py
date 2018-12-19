# -*- coding: utf-8 -*-

import numpy as np
import ofrandom as ofr
from copy import deepcopy
from ofnumber import OFNumber, flog
from scipy.optimize import minimize


class OFSeries(object):
    def __init__(self, ofns):
        super(OFSeries, self).__init__()
        self.values = np.array(ofns, dtype=object)
        
    def copy(self):
        return deepcopy(self)
    
    def __getitem__(self, i):
        return self.values[i]
    
    def __setitem__(self, i, ofn):
        self.values[i] = ofn
        
    def plot_ofseries(self, ax, s=0, e=None, color='black', shift=0, ord_method='expected'):
        if e is None:
            ofns = self[s:]
        else:
            ofns = self[s:e]
        for i, ofn in enumerate(ofns):
            f, g = ofn.branch_f, ofn.branch_g
            x = f.domain_x*0.5 + i-0.25 + shift
            o = ofn.order(method=ord_method)
            if o >= 0:
                ax.fill_between(x, f.fvalue_y, g.fvalue_y, facecolor='white', edgecolor='black')
            else:
                ax.fill_between(x, f.fvalue_y, g.fvalue_y, facecolor=color, edgecolor='black')
        ax.set_xlim(-1+shift, len(ofns)+shift)
        
    def to_positive_order(self, method='expected', args=()):
        fv = np.vectorize(lambda x: x if x.order(method=method, args=args) >= 0.0 else x.change_order(), otypes=[OFNumber])
        return OFSeries(fv(self.values))
        
    def to_array(self, stack='vstack'): 
        fv = np.vectorize(lambda x: x.to_array(stack=stack), otypes=[np.ndarray])
        return fv(self.values)
    
    def defuzzy(self, method='scog', args=(0.5,)):
        fv = np.vectorize(lambda x: x.defuzzy(method=method, args=args), otypes=[np.double])
        return fv(self.values)
    
    def order(self, method='scog', args=(0.5,)):
        fv = np.vectorize(lambda x: x.order(method=method, args=args), otypes=[np.int])
        return fv(self.values)
    
    def mean_fuzzy(self):
        x = np.mean(self.to_array())
        return OFNumber(x[0], x[1])
    
    def mean_crisp(self):
        mu = self.mean_fuzzy()
        return mu.defuzzy(method='expected')
    
    def var_fuzzy(self, ddof=1):
        x = np.var(self.to_array(), ddof=ddof)
        return OFNumber(x[0], x[1])
    
    def var_crisp(self, ddof=1):
        defuzz = self.defuzzy(method='expected')
        return np.var(defuzz, ddof=ddof)
    
    def order_probability(self):
        ords = self.order(method='expected')
        return ords[ords >= 0].sum()/ords.shape[0]
    
    def plot_histogram(self, ax_f, ax_g, alpha, bins=20, density=False, s=0, e=None, kwargs_f={}, kwargs_g={}):
        if e is None:
            ofns = self[s:]
        else:
            ofns = self[s:e]
        fv_f = np.vectorize(lambda x: x.branch_f(alpha), otypes=[np.double])
        fv_g = np.vectorize(lambda x: x.branch_g(alpha), otypes=[np.double])
        data_f = fv_f(self.values)
        data_g = fv_g(self.values)
        ax_f.hist(data_f, bins=bins, density=density, **kwargs_f)
        ax_g.hist(data_g, bins=bins, density=density, **kwargs_g)

#TODO: Dokończyć
    def plot_3d_histogram(self, ax_f, ax_g, alpha=np.linspace(0, 1, 11), bins=20, density=False, s=0, e=None, kwargs_f={}, kwargs_g={}):
        if e is None:
            ofns = self[s:]
        else:
            ofns = self[s:e]


        
    def transform(self, method='diff'):
        arr = np.copy(self.values)
        if method == 'diff':
            new_ofns = arr[1:]-arr[:-1]
        elif method == 'ret':
            new_ofns = (arr[1:]-arr[:-1])/arr[:-1]
        elif method == 'logret':
            fv = np.vectorize(lambda x: flog(x), otypes=[OFNumber])
            new_ofns = fv(arr[1:]/arr[:-1])
        else:
            raise ValueError('method must be diff, ret or logret')
        return OFSeries(new_ofns) 


# TODO: Przerobić bez pętli, przetestowac wersję z wykorzystaniem tensorflow
class OFAutoRegressive(object):
    def __init__(self, order=1, intercept=True, coef=[], initial=[]):
        super(OFAutoRegressive, self).__init__()
        self.intercept = intercept
        self.order = order
        self.coef = OFSeries(coef)
        self.initial = initial
        self.residuals = None
        
    def fit(self, ofseries, order, intercept=True, method='ls', solver='L-BFGS-B', options={}):
        dim = ofseries[0].branch_f.dim
        self.order = order
        self.intercept = intercept
        self.initial = ofseries[-order-1:]
        
        n_coef = order
        if self.intercept:
            n_coef += 1

        # initial coef
        coef = OFSeries(ofr.ofnormal_sample(n_coef, OFNumber.init_from_scalar(0.0, dim=dim), OFNumber.init_from_scalar(0.001, dim=dim), 1, 0.5))

        if solver == 'L-BFGS-B':
            if options == {}:
                options = {'disp': None, 'gtol': 1.0e-12, 'eps': 1e-08, 'maxiter': 1000, 'ftol': 2.22e-09}
            p0 = np.concatenate(coef.to_array(stack='hstack'))
            #print(p0)
            ofns = np.concatenate(ofseries.to_array(stack='hstack'))
            args = (order, n_coef, dim, ofns, intercept)
            if method == 'ls':
                res = minimize(fun_obj_ols, p0, args=args, method='L-BFGS-B', jac=True, options=options)
                coef = array2ofns(res.x, n_coef, dim)
            else:
                raise ValueError('wrong method')
            self.coef = OFSeries(coef)
        else:
            raise ValueError('wrong solver')

        residuals = []
        for i in range(order, len(ofseries.values)):
            if i < order:
                residuals.append(OFNumber(np.zeros(dim), np.zeros(dim)))
            else:
                pred = self.predict(1, initial=list(ofseries[i-order:i]))
                residuals.append(ofseries[i]-pred[0])
        self.residuals = OFSeries(residuals)

    def predict(self, n, initial=None, mean=None):
        if initial is None:
            initial = self.initial
        predicted = []
        for t in range(1, n+1):
            if self.intercept:
                y = self.coef[0]
                for p in range(1, self.order+1):
                    y = y + self.coef[p] * initial[-p]
            else:
                y = self.coef[0] * initial[-1]
                for p in range(1, self.order):
                    y = y + self.coef[p] * initial[-p-1]
            if mean is not None:
                y = y + mean
            predicted.append(y)
            initial.append(y)
        return OFSeries(predicted)    
    
    
def fun_obj_ols(p, order, n_coef, dim, ofns, intercept):
    e = 0.0
    n_cans = int(len(ofns)/(2 * dim))
    can = ofns.reshape((n_cans, 2 * dim))
    coef = p.reshape((n_coef, 2 * dim))
    grad = np.zeros(len(p))
    if intercept:
        for i in range(order, n_cans):
            r = can[i] - autoreg_bias(coef, can[i-order:i])
            e += np.sum(r * r)
            grad[:2 * dim] -= 2.0 * r
            for j in range(1, n_coef):
                grad[2 * dim * j:2 * dim * (j + 1)] -= 2 * r * can[i - j]
    else:
        for i in range(n_coef, n_cans):
            r = can[i] - autoreg_unbias(coef, can[i-n_coef:i])
            e += np.sum(r * r)
            for j in range(n_coef):
                grad[2 * dim * j:2 * dim * (j + 1)] -= 2 * r * can[i - j-1]
    return e, grad


def array2ofns(arr, n, dim):
    ofns = []
    for nc in range(n):
        s1 = nc*2*dim
        s2 = nc*2*dim + dim
        ofns.append(OFNumber(arr[s1:s2], arr[s2:s2+dim]))
    return np.array(ofns, dtype=object)


def autoreg_bias(coef, past):
    y = coef[0].copy()
    for p in range(1, len(coef)):
        y = y + coef[p] * past[-p]
    return y


def autoreg_unbias(coef, past):
    y = coef[0] * past[-1]
    for p in range(1, len(coef)):
        y = y + coef[p] * past[-p-1]
    return y
