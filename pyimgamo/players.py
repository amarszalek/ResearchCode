import numpy as np
from utils import  population_suppression
from operators import create_individual_uniform, hiper_mutate
import copy


# base class for player
class Player(object):
    def __init__(self, player_id):
        self.player_id = player_id

    def create_individual(self, nvars, bounds):
        return create_individual_uniform(nvars, bounds)

    def create_population(self, npop, nvars, bounds):
        pop = np.zeros((npop, nvars))
        for i in range(npop):
            pop[i] = self.create_individual(nvars, bounds)
        return pop

    def optimize(self, solutions, solutions_eval, pattern, bounds):
        evaluation_count = np.zeros(solutions_eval.shape[1])
        new_solutions = solutions.copy()
        new_solutions_eval = solutions_eval.copy()
        return new_solutions, new_solutions_eval, evaluation_count


class ClonalSelection(Player):
    def __init__(self, player_id, nclone=15, supp_level=0, mutate=hiper_mutate, mutate_args=(0.45, 0.9, 0.1)):
        super(ClonalSelection, self).__init__(player_id)
        self.nclone = nclone
        self.supp_level = supp_level
        self.mutate = mutate
        self.mutare_args = mutate_args

    def optimize(self, solutions, solutions_eval, pattern, problem):
        evaluation_count = np.zeros(solutions_eval.shape[1])
        new_solutions = copy.deepcopy(solutions)
        new_solutions_eval = copy.deepcopy(solutions_eval)
        temp_pop_eval = copy.deepcopy(solutions_eval[:, self.player_id])
        arg_sort = temp_pop_eval.argsort()
        indices = []
        better = []
        clone_num = self.nclone
        for arg in arg_sort:
            clones = np.array([self.mutate(copy.deepcopy(solutions[arg]), pattern, problem.bounds, *self.mutare_args)
                               for _ in range(clone_num)])
            if problem.need_repair:
                clones = problem.repair(clones)
            clones_eval = problem.evaluate_one(clones, self.player_id)
            evaluation_count[self.player_id] += clone_num
            argmin = clones_eval.argmin()
            if clones_eval[argmin] < solutions_eval[arg, self.player_id]:
                indices.append(arg)
                better.append(clones[argmin])
            clone_num = clone_num - 1 if clone_num > 2 else 1
        if len(better) > 0:
            better = np.stack(better)
            better_eval = problem.evaluate_all(better)
            evaluation_count += np.size(better_eval)
            new_solutions[indices] = better
            new_solutions_eval[indices] = better_eval
        # suppression
        if self.supp_level > 0:
            mask = population_suppression(new_solutions, self.supp_level)
            new = self.create_population(np.sum(mask), problem.nvars, problem.bounds)
            if problem.need_repair:
                new = problem.repair(new)
            new_eval = problem.evaluate_all(new)
            evaluation_count += np.size(new_eval)
            new_solutions[mask] = new
            new_solutions_eval[mask] = new_eval
        return new_solutions, new_solutions_eval, evaluation_count
