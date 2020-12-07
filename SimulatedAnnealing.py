import typing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def distance(x: np.ndarray, y: np.ndarray) -> float:
    return np.linalg.norm(x - y)


def mean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([(x[0] + y[0]) / 2, (x[1] + y[1]) / 2])


class Solution(object):
    def __init__(self, lmbd: float, dataset) -> None:
        self._lmbd = lmbd
        self._dataset = dataset
        self._c_i: typing.Optional[int] = None
        self._c_j: typing.Optional[int] = None
        self._center: typing.Optional[np.ndarray] = None
        self._radius: float = 0
        self.assignments = (np.zeros(self._dataset.N) == 1)
        self._objective_v: float = 0

    def incremental_objective_function(self, move: int) -> None:
        self.assignments[move] = not self.assignments[move]
        old_radius = self._radius
        if self.assignments[move]:  # old_assignment = False
            self._objective_v += self._dataset.v[move]
            changed_flag: bool = False
            self._radius = 0  # TODO: not needed? if yes, the else on the changed_flag is not needed?
            # TODO: iterate only over assigned cities (move included) -> linear
            for j in range(self._dataset.N):
                if self.assignments[j]:
                    new_radius = self.index_distance(move, j) / 2
                    if new_radius >= self._radius and new_radius >= old_radius:
                        changed_flag = True
                        self._c_i = move
                        self._c_j = j
                        self._radius = new_radius
            if changed_flag:
                self._center = self.index_mean(self._c_i, self._c_j)
                # self.objective += self._lmbd * self._dataset.N * np.pi * (
                #         np.power(old_radius, 2) - np.power(self._radius, 2))
            else:
                self._radius = old_radius
        else:  # old_assignment = True
            self._objective_v -= self._dataset.v[move]
            if move == self._c_i or move == self._c_j:
                self._radius = 0
                # TODO: iterate only over assigned cities -> quadratic (better??)
                for i in range(self._dataset.N):
                    if self.assignments[i]:
                        for j in range(self._dataset.N):
                            if self.assignments[j]:
                                new_radius = self.index_distance(i, j) / 2
                                if new_radius > self._radius:
                                    self._c_i = i
                                    self._c_j = j
                                    self._radius = new_radius
                if move == self._c_i and move == self._c_j:
                    self._c_i = None
                    self._c_j = None
                    self._center = None
                elif move == self._c_i:
                    self._c_i = self._c_j
                    self._center = self._dataset.x[self._c_j]
                elif move == self._c_j:
                    self._c_j = self._c_i
                    self._center = self._dataset.x[self._c_i]
                else:
                    self._center = self.index_mean(self._c_i, self._c_j)
                # self.objective += self._lmbd * self._dataset.N * np.pi * (
                #         np.power(old_radius, 2) - np.power(self._radius, 2))

    def get_objective(self) -> float:
        return self._objective_v - self._lmbd * self._dataset.N * np.pi * np.power(self._radius, 2)

    def plot(self) -> None:
        a = sns.relplot(self._dataset.x[:, 0], self._dataset.x[:, 1], hue=self.assignments, size=self._dataset.v)
        if self._center is not None:
            a.ax.add_artist(plt.Circle(tuple(self._center), self._radius, color='black', fill=False))
            a.ax.add_artist(
                plt.Circle(self._dataset.x[self._c_i], 2 * self._radius, color='grey', linestyle='--', fill=False))
            a.ax.add_artist(
                plt.Circle(self._dataset.x[self._c_j], 2 * self._radius, color='grey', linestyle='--', fill=False))
        plt.title("s = " + str(np.round(self._objective_v, 2)) + " ; r = " + str(np.round(self._radius, 2)))
        plt.show()

    def index_distance(self, i: int, j: int) -> float:
        return distance(self._dataset.x[i], self._dataset.x[j])

    def index_mean(self, i: int, j: int) -> np.ndarray:
        return mean(self._dataset.x[i], self._dataset.x[j])


class SimulatedAnnealing(object):
    def __init__(self, lmbd: float, dataset, p: float):
        self._lmbd = lmbd
        self._dataset = dataset
        self._p = p
        self.S = self._initial_solution()
        self.S.plot()
        self.objectives = [self.S.get_objective()]
        self._beta = self._heat_up()
        print(self._beta)

    def _initial_solution(self) -> Solution:
        initial_solution: Solution = Solution(self._lmbd, self._dataset)
        # for i in np.random.randint(self._dataset.N, size=int(self._dataset.N / 2)):
        for i in [np.random.randint(self._dataset.N)]:
            initial_solution.incremental_objective_function(i)
        return initial_solution

    def _heat_up(self) -> float:
        old_obj = self.S.get_objective()
        sum_obj = 0
        count = 0
        for i in range(self._dataset.N):
            # TODO
            self.S.incremental_objective_function(i)
            new_obj = self.S.get_objective()
            if new_obj < old_obj:
                sum_obj += new_obj
                count += 1
            self.S.incremental_objective_function(i)
        return np.log(self._p) / (sum_obj / count - self.S.get_objective())

    def cool_down(self, iters: int) -> None:  # TODO
        for i in tqdm(range(iters)):
            old_objective = self.S.get_objective()
            move = np.random.randint(self._dataset.N)
            self.S.incremental_objective_function(move)
            if np.random.random() > np.exp(self._beta * (self.S.get_objective() - old_objective)):
                self.S.incremental_objective_function(move)
            self.objectives.append(self.S.get_objective())
        self.S.plot()
