# -*- coding: utf-8 -*-

import numpy as np
import ofmodels as ofm
from ofnumber import OFNumber, fmax


# generate peudo ordered fuzzy random variable with normal distributon
def ofnormal(mu, sig2, s2, p):
    minv = min(np.min(mu.branch_f.fvalue_y), np.min(mu.branch_g.fvalue_y))
    c = np.abs(minv) + 1.0 if minv <= 0.0 else 0.0
    eta = mu + c
    x = np.random.normal(1.0, np.sqrt(sig2.branch_f.fvalue_y) / eta.branch_f.fvalue_y)
    y = np.random.normal(1.0, np.sqrt(sig2.branch_g.fvalue_y) / eta.branch_g.fvalue_y)
    if np.random.random() < p:
        ksi = OFNumber(x, y) * eta + np.random.normal(0, np.sqrt(s2))
    else:
        ksi = OFNumber(x, y) * eta.change_order() + np.random.normal(0, np.sqrt(s2))
    return ksi - c


def ofnormal_sample(n, mu, sig2, s2, p):
    minv = min(np.min(mu.branch_f.fvalue_y), np.min(mu.branch_g.fvalue_y))
    c = np.abs(minv) + 1.0 if minv <= 0.0 else 0.0
    eta = mu + c
    dim = eta.branch_f.dim
    sig2_f = np.tile(sig2.branch_f.fvalue_y, (n, 1))
    sig2_g = np.tile(sig2.branch_g.fvalue_y, (n, 1))
    x = np.random.normal(1.0, np.sqrt(sig2_f) / eta.branch_f.fvalue_y)
    y = np.random.normal(1.0, np.sqrt(sig2_g) / eta.branch_g.fvalue_y)
    s = np.tile(np.random.normal(0, np.sqrt(s2), n), (dim, 1)).T
    r = np.tile(np.random.random(n), (dim, 1)).T
    ksi_x = np.where(r < p, x * eta.branch_f.fvalue_y + s, x * eta.branch_g.fvalue_y + s)
    ksi_y = np.where(r < p, y * eta.branch_g.fvalue_y + s, y * eta.branch_f.fvalue_y + s)
    ksi = np.hstack([ksi_x, ksi_y]) - c
    return ofm.OFSeries(np.apply_along_axis(lambda xx: OFNumber(xx[:dim], xx[dim:]), 1, ksi))


def ofnormal_mu_est(ofs):
    pofs = ofs.to_positive_order()
    return pofs.mean_fuzzy()


def ofnormal_sig2_est(ofs, ddof=1):
    s2 = ofnormal_s2_est(ofs, ddof=ddof)
    pofs = ofs.to_positive_order()
    return fmax(pofs.var_fuzzy(ddof=ddof)-s2, 0.0)


def ofnormal_s2_est(ofs, ddof=1):
    return ofs.var_crisp(ddof=ddof)
    

def ofnormal_p_est(ofs):
    return ofs.order_probability()


# generate peudo ordered fuzzy random variable with uniform distributon
def ofuniform(mu, sig2, s2, p):
    s_f = np.random.uniform(-np.sqrt(3*sig2.branch_f.fvalue_y), np.sqrt(3*sig2.branch_f.fvalue_y))
    s_g = np.random.uniform(-np.sqrt(3*sig2.branch_g.fvalue_y), np.sqrt(3*sig2.branch_g.fvalue_y))
    s = np.random.uniform(-np.sqrt(3*s2), np.sqrt(3*s2))
    x = mu.branch_f.fvalue_y + s_f + s
    y = mu.branch_g.fvalue_y + s_g + s
    if np.random.random() < p:
        return OFNumber(x, y)
    else:
        return OFNumber(y, x)


def ofuniform_sample(n, mu, sig2, s2, p):
    sig2_f = np.tile(sig2.branch_f.fvalue_y, (n, 1))
    sig2_g = np.tile(sig2.branch_g.fvalue_y, (n, 1))
    s_f = np.random.uniform(-np.sqrt(3*sig2_f), np.sqrt(3*sig2_f))
    s_g = np.random.uniform(-np.sqrt(3*sig2_g), np.sqrt(3*sig2_g))
    dim = mu.branch_f.dim
    s = np.tile(np.random.uniform(-np.sqrt(3*s2), np.sqrt(3*s2), n), (dim, 1)).T
    r = np.tile(np.random.random(n), (dim, 1)).T
    ksi_x = np.where(r < p, mu.branch_f.fvalue_y + s_f + s, mu.branch_g.fvalue_y + s_g + s)
    ksi_y = np.where(r < p, mu.branch_g.fvalue_y + s_g + s, mu.branch_f.fvalue_y + s_f + s)
    ksi = np.hstack([ksi_x, ksi_y])
    return ofm.OFSeries(np.apply_along_axis(lambda x: OFNumber(x[:dim], x[dim:]), 1, ksi))

    
def ofuniform_mu_est(ofs):
    pofs = ofs.to_positive_order()
    return pofs.mean_fuzzy()
    
    
def ofuniform_sig2_est(ofs, ddof=1):
    s2 = ofnormal_s2_est(ofs, ddof=ddof)
    pofs = ofs.to_positive_order()
    return fmax(pofs.var_fuzzy(ddof=ddof)-s2, 0.0)
    
    
def ofuniform_s2_est(ofs, ddof=1):
    return ofs.var_crisp(ddof=ddof)
    
    
def ofuniform_p_est(ofs):
    return ofs.order_probability()
