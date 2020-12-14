import typing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.spatial.distance as dist
import copy
import helpers


class Solution(object):
    def __init__(self, lmbd: float, dataset, order: np.ndarray) -> None:
        self._lmbd = lmbd
        self._dataset = dataset
        self._order = order
        self._c_i: typing.Optional[int] = 0
        self._c_j: typing.Optional[int] = dataset.N - 1
        self._center: typing.Optional[np.ndarray] = np.array([0, 0])
        self._radius: float = 0
        self._assignments = np.zeros(self._dataset.N) == 1
        self._objective_v: float = 0
        self.update()

    def update(self) -> None:

        self._radius = self.index_distance(self._order[self._c_i], self._order[self._c_j]) / 2
        self._center = self.index_mean(self._order[self._c_i], self._order[self._c_j])

        for j in range(self._dataset.N):
            dist_ = helpers.distance(self._dataset.x[j], self._center)

            if self._assignments[j]:
                if dist_ > self._radius and j != self._order[self._c_i] and j != self._order[self._c_j]:
                    self._assignments[j] = not self._assignments[j]
                    self._objective_v -= self._dataset.v[j]
            else:
                if dist_ <= self._radius or j == self._order[self._c_i] or j == self._order[self._c_j]:
                    self._assignments[j] = not self._assignments[j]
                    self._objective_v += self._dataset.v[j]

    def move(self, select_move: typing.Optional[int] = None, select_point: typing.Optional[int] = None) -> None:

        if select_point is None:
            point_to_replace = np.random.choice((self._c_i, self._c_j))
        elif select_point == 0:
            point_to_replace = self._c_i
        else:
            point_to_replace = self._c_j

        if select_move is None:
            move = np.random.choice((-1, 1))
        else:
            move = select_move

        if point_to_replace == self._c_i:
            if (not (self._c_i == 0 and move == -1)) and (not (self._c_i == self._dataset.N - 1 and move == 1)):
                self._c_i += move
        else:
            if (not (self._c_j == 0 and move == -1)) and (not (self._c_j == self._dataset.N - 1 and move == 1)):
                self._c_j += move

        if self._c_i == self._c_j:
            self._radius = 0
            self._center = self._dataset.x[self._order[self._c_i]]
            self._assignments = np.zeros(self._dataset.N) == 1
            self._assignments[self._order[self._c_i]] = True
            self._objective_v = self._dataset.v[self._order[self._c_i]]

        else:
            self.update()

    def get_objective(self) -> float:
        return self._objective_v - self._lmbd * self._dataset.N * np.pi * np.power(self._radius, 2)

    def get_profit(self) -> float:
        return self._objective_v

    def get_cost(self) -> float:
        return self._dataset.N * np.pi * np.power(self._radius, 2)

    def plot(self) -> None:
        a = sns.relplot(self._dataset.x[:, 0], self._dataset.x[:, 1], hue=self._assignments, size=self._dataset.v)
        if self._center is not None:
            a.ax.add_artist(plt.Circle(tuple(self._center), self._radius, color='black', fill=False))
            a.ax.add_artist(
                plt.Circle(self._dataset.x[self._order[self._c_i]], 2 * self._radius, color='grey', linestyle='--', fill=False))
            a.ax.add_artist(
                plt.Circle(self._dataset.x[self._order[self._c_j]], 2 * self._radius, color='grey', linestyle='--', fill=False))
        plt.title("s = " + str(np.round(self._objective_v, 2)) + " ; r = " + str(np.round(self._radius, 2)))
        plt.show()

    def index_distance(self, i: int, j: int) -> float:
        return helpers.distance(self._dataset.x[i], self._dataset.x[j])

    def index_assigned_distances(self, assigned):
        assigned_points = self._dataset.x[assigned]
        return dist.pdist(assigned_points)

    def index_mean(self, i: int, j: int) -> np.ndarray:
        return helpers.mean(self._dataset.x[i], self._dataset.x[j])


class SimulatedAnnealing(object):
    def __init__(self, lmbd: float, dataset, order: np.ndarray, alpha: float = 0.5):
        self._lmbd = lmbd
        self._dataset = dataset
        self._order = order
        # self._p = p
        self.S = self._initial_solution()
        self.S.plot()
        self.objectives = [self.S.get_objective()]
        self._beta = 0
        self.betas = []
        self._best_obj = self.S.get_objective()
        self._alpha = alpha
        # print(self._beta)

    def _initial_solution(self) -> Solution:
        initial_solution: Solution = Solution(self._lmbd, self._dataset, self._order)
        return initial_solution

    def heat_cool_cycles(self, iters: int, cycles: int):

        p_range = np.linspace(0, 1, cycles + 2)

        for i in range(cycles):
            print("Heat up cycle %d " % (i + 1))
            self._beta = self._heat_up(p_range[cycles - i])
            print("Current p %f" % p_range[cycles - i])
            print("Beta : %f " % self._beta)
            print("Objective init %f " % self.S.get_objective())
            self.cool_down(iters)
            print("Objective final %f " % self.S.get_objective())

    def _heat_up(self, p: float) -> float:
        old_obj = self._best_obj
        sum_obj = 0
        count = 0
        for i in [-1, 1]:
            # TODO
            for elem in [0, 1]:
                s_prime = copy.deepcopy(self.S)
                s_prime.move(i, elem)
                new_obj = s_prime.get_objective()
                if new_obj < old_obj:
                    sum_obj += new_obj
                    count += 1
        if count == 0:
            return self._beta / self._alpha
        mean_ = sum_obj / count
        # p: Probability with beta to choose the average worsening solution
        self.betas.append(np.log(p) / (mean_ - old_obj))
        return np.max((self._beta / self._alpha, np.log(p) / (mean_ - old_obj)))

    def cool_down(self, iters: int) -> None:  # TODO

        for i in tqdm(range(iters)):
            old_objective = self.S.get_objective()
            s_prime = copy.deepcopy(self.S)
            s_prime.move()
            if np.random.random() <= np.exp(self._beta * (s_prime.get_objective() - old_objective)):
                self.S = s_prime
                current_objective = self.S.get_objective()
                if current_objective > self._best_obj:
                    self._best_obj = current_objective
            self.objectives.append(self.S.get_objective())
        self.S.plot()

    def plot_betas(self) -> None:
        plt.plot(range(len(self.betas)), self.betas)
        plt.show()
