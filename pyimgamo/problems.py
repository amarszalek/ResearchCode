import numpy as np


# base class for problem
class Problem(object):
    def __init__(self, nvars, nobjs, bounds, need_repair=True):
        super(Problem, self).__init__()
        self.nvars = nvars
        self.nobjs = nobjs
        self.bounds = bounds
        self.need_repair = need_repair

    # evaluate all objective functions
    def evaluate_all(self, solutions):
        pass

    # evaluate selected objective function
    def evaluate_one(self, solutions, i):
        pass

    def repair(self, solutions):
        return solutions


class Kursawe(Problem):
    def __init__(self, nvars=3, nobjs=2, bounds=((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)), need_repair=False):
        super(Kursawe, self).__init__(nvars, nobjs, bounds, need_repair)

    def evaluate_all(self, solutions):
        f = np.sum(-10.0 * np.exp(-0.2 * np.sqrt(solutions[:, :-1]**2 + solutions[:, 1:]**2)), axis=1)
        g = np.sum(np.abs(solutions)**0.8 + 5.0 * np.sin(solutions**3), axis=1)
        return np.stack([f, g], axis=1)

    def evaluate_one(self, solutions, i):
        if i == 0:
            return np.sum(-10.0 * np.exp(-0.2 * np.sqrt(solutions[:, :-1]**2 + solutions[:, 1:]**2)), axis=1)
        if i == 1:
            return np.sum(np.abs(solutions)**0.8 + 5.0 * np.sin(solutions**3), axis=1)
