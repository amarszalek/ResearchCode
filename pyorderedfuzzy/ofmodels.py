# -*- coding: utf-8 -*-

import numpy as np
import ofrandom as ofr
from copy import deepcopy
from ofnumber import OFNumber, flog
from scipy.optimize import minimize
from collections import deque


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

    def __len__(self):
        return len(self.values)
        
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
        fv = np.vectorize(lambda x: x if x.order(method=method, args=args) >= 0.0 else x.change_order(),
                          otypes=[OFNumber])
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
    
    def plot_histogram(self, ax_f, ax_g, alpha, bins=20, density=False, s=0, e=None, kwargs_f=None, kwargs_g=None):
        if kwargs_f is None:
            kwargs_f = {}
        if kwargs_g is None:
            kwargs_g = {}
        if e is None:
            ofns = self.values[s:]
        else:
            ofns = self.values[s:e]
        fv_f = np.vectorize(lambda x: x.branch_f(alpha), otypes=[np.double])
        fv_g = np.vectorize(lambda x: x.branch_g(alpha), otypes=[np.double])
        data_f = fv_f(ofns)
        data_g = fv_g(ofns)
        ax_f.hist(data_f, bins=bins, density=density, **kwargs_f)
        ax_g.hist(data_g, bins=bins, density=density, **kwargs_g)

    def plot_3d_histogram(self, ax_f, ax_g, alphas=np.linspace(0, 1, 11), bins=20, density=False, s=0, e=None,
                          kwargs_f=None, kwargs_g=None):
        if kwargs_f is None:
            kwargs_f = {}
        if kwargs_g is None:
            kwargs_g = {}
        if e is None:
            ofns = self.values[s:]
        else:
            ofns = self.values[s:e]
        for a in alphas:
            fv_f = np.vectorize(lambda x: x.branch_f(a), otypes=[np.double])
            fv_g = np.vectorize(lambda x: x.branch_g(a), otypes=[np.double])
            data_f = fv_f(ofns)
            data_g = fv_g(ofns)
            h_f, b_f = np.histogram(data_f, bins=bins, density=density)
            b_f = (b_f[:-1] + b_f[1:]) / 2.
            h_g, b_g = np.histogram(data_g, bins=bins, density=density)
            b_g = (b_g[:-1] + b_g[1:]) / 2.
            ax_f.bar(b_f, h_f, zs=a, zdir='y', alpha=0.8, **kwargs_f)
            ax_g.bar(b_g, h_g, zs=a, zdir='y', alpha=0.8, **kwargs_g)
            ax_f.set_xlabel('$f(\\alpha)$')
            ax_f.set_ylabel('$\\alpha$')
            ax_f.set_zlabel('frequency')
            ax_g.set_xlabel('$g(\\alpha)$')
            ax_g.set_ylabel('$\\alpha$')
            ax_g.set_zlabel('frequency')
        
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


class OFAutoRegressive(object):
    def __init__(self, order=1, intercept=True):
        super(OFAutoRegressive, self).__init__()
        self.intercept = intercept
        self.order = order

    def fit(self, ofseries, method='ls', solver='L-BFGS-B', options=None):
        dim = ofseries[0].branch_f.dim
        self.initials = ofseries[-self.order - 1:]

        # initial coef
        n_coef = self.order + 1 if self.intercept else self.order
        coef = ofr.ofnormal_sample(n_coef, OFNumber.init_from_scalar(0.0, dim=dim),
                                   OFNumber.init_from_scalar(0.001, dim=dim), 1, 0.5)

        if solver == 'L-BFGS-B':
            if options is None:
                options = {'disp': None, 'gtol': 1.0e-12, 'eps': 1e-08, 'maxiter': 1000, 'ftol': 2.22e-09}
            p0 = np.concatenate(coef.to_array(stack='hstack'))
            ofns = np.stack(ofseries.to_array(stack='hstack'))
            y = ofns[self.order:]
            x = np.ones((len(ofseries) - self.order, n_coef, 2*dim))
            for i in range(self.order, len(ofseries)):
                if self.intercept:
                    x[i - self.order, 1:, :] = (ofns[i-self.order:i][:])[::-1]
                else:
                    x[i-self.order] = (ofns[i-self.order:i])[::-1]

            args = (n_coef, dim, x, y)
            if method == 'ls':
                res = minimize(lr_fun_obj_ols, p0, args=args, method='L-BFGS-B', jac=True, options=options)
                coef = array2ofns(res.x, n_coef, dim)
            else:
                raise ValueError('wrong method')
            self.coefs = OFSeries(coef)
        else:
            raise ValueError('wrong solver')

        residuals = []
        for i in range(self.order, len(ofseries)):
            pred = self.predict(1, initials=list(ofseries[i - self.order:i]))
            residuals.append(ofseries[i] - pred[0])
        self.residuals = OFSeries(residuals)

    def predict(self, n, coefs=None, initials=None, mean=None, error=False, er_opt=None):
        if coefs is None:
            if hasattr(self, 'coefs'):
                coefs = deepcopy(self.coefs)
            else:
                raise ValueError('No attribute coefs')
        if initials is None:
            if hasattr(self, 'initials'):
                initials = deque(deepcopy(self.initials), maxlen=len(self.initials))
            else:
                raise ValueError('No attribute initials')
        else:
            initials = deque(initials, maxlen=len(initials))

        dim = coefs[0].branch_f.dim
        er = None
        if error:
            if er_opt is None:
                er_opt = {'dist': 'ofnormal', 'mu': OFNumber.init_from_scalar(0.0, dim=dim),
                          'sig2': OFNumber.init_from_scalar(0.0, dim=dim), 's2': 1, 'p': 0.5}
            if er_opt['dist'] == 'ofnormal':
                er = ofr.ofnormal_sample(n, er_opt['mu'], er_opt['sig2'], er_opt['s2'], er_opt['p'])
        else:
            er = OFSeries([OFNumber.init_from_scalar(0.0, dim=dim) for _ in range(n)])

        predicted = []
        for t in range(n):
            if self.intercept:
                y = coefs[0]
                for p in range(1, self.order+1):
                    y = y + coefs[p] * initials[-p]
            else:
                y = coefs[0] * initials[-1]
                for p in range(1, self.order):
                    y = y + coefs[p] * initials[-p-1]
            if mean is not None:
                y = y + mean
            y = y + er[t]
            predicted.append(y)
            initials.append(y)
        return OFSeries(predicted)


class OFLinearRegression(object):
    def __init__(self, intercept=True):
        super(OFLinearRegression, self).__init__()
        self.intercept = intercept

    # x, y -> np.array of OFN
    # x.shape = (T, N), y.shape = (T,)
    def fit(self, x, y, solver='L-BFGS-B', options=None):
        dim = x[0, 0].branch_f.dim

        # initial coef
        n_coef = x.shape[1] + 1 if self.intercept else x.shape[1]
        coef = ofr.ofnormal_sample(n_coef, OFNumber.init_from_scalar(0.0, dim=dim),
                                   OFNumber.init_from_scalar(0.001, dim=dim), 1, 0.5)

        if solver == 'L-BFGS-B':
            if options is None:
                options = {'disp': None, 'gtol': 1.0e-12, 'eps': 1e-08, 'maxiter': 1000, 'ftol': 2.22e-09}
            p0 = np.concatenate(coef.to_array(stack='hstack'))
            yy = np.stack(OFSeries(y).to_array(stack='hstack'))
            xx = np.ones((len(y), n_coef, 2*dim))
            for i in range(len(y)):
                if self.intercept:
                    xx[i, 1:, :] = np.stack(OFSeries(x[i]).to_array(stack='hstack'))
                else:
                    xx[i] = np.stack(OFSeries(x[i]).to_array(stack='hstack'))

            args = (n_coef, dim, xx, yy)
            res = minimize(lr_fun_obj_ols, p0, args=args, method='L-BFGS-B', jac=True, options=options)
            coef = array2ofns(res.x, n_coef, dim)
            self.coefs = OFSeries(coef)
        else:
            raise ValueError('wrong solver')

        pred = self.predict(x)
        self.residuals = OFSeries(y - pred.values)

    def predict(self, x, error=False, er_opt=None):
        dim = self.coefs[0].branch_f.dim
        n = x.shape[0]
        er = None
        if error:
            if er_opt is None:
                er_opt = {'dist': 'ofnormal', 'mu': OFNumber.init_from_scalar(0.0, dim=dim),
                          'sig2': OFNumber.init_from_scalar(0.0, dim=dim), 's2': 1, 'p': 0.5}
            if er_opt['dist'] == 'ofnormal':
                er = ofr.ofnormal_sample(n, er_opt['mu'], er_opt['sig2'], er_opt['s2'], er_opt['p'])
        else:
            er = OFSeries([OFNumber.init_from_scalar(0.0, dim=dim) for _ in range(n)])

        if self.intercept:
            pred = np.sum(x * self.coefs[1:], axis=1) + self.coefs[0] + er.values
        else:
            pred = np.sum(x * self.coefs, axis=1) + er.values
        return OFSeries(pred)


def lr_fun_obj_ols(p, n_coef, dim, x, y):
    coef = p.reshape((n_coef, 2 * dim))
    yp = np.sum(x*coef, axis=1)
    e = np.sum(np.power(y - yp, 2))
    g = [np.sum(2*(y-yp)*x[:, i, :], axis=0) for i in range(x.shape[1])]
    grad = np.concatenate(g)
    return e, -grad


def array2ofns(arr, n, dim):
    ofns = []
    for nc in range(n):
        s1 = nc*2*dim
        s2 = nc*2*dim + dim
        ofns.append(OFNumber(arr[s1:s2], arr[s2:s2+dim]))
    return np.array(ofns, dtype=object)
