from dataclasses import dataclass

import numpy as np
from evobench import Benchmark, Population
from lazy import lazy


@dataclass
class Data:

    benchmark: Benchmark
    pop_size_train: int
    pop_size_val: int
    pop_size_test: int

    @lazy
    def pop_train(self) -> Population:
        pop_train = self.benchmark.initialize_population(self.pop_size_train)
        self.benchmark.evaluate_population(pop_train)

        return pop_train

    @lazy
    def pop_val(self) -> Population:
        pop_val = self.benchmark.initialize_population(self.pop_size_val)
        self.benchmark.evaluate_population(pop_val)

        return pop_val

    @lazy
    def pop_test(self) -> Population:
        pop_test = self.benchmark.initialize_population(self.pop_size_test)
        self.benchmark.evaluate_population(pop_test)

        return pop_test

    @lazy
    def x_train(self) -> np.ndarray:
        return self.pop_train.as_ndarray

    @lazy
    def x_val(self) -> np.ndarray:
        return self.pop_val.as_ndarray

    @lazy
    def x_test(self) -> np.ndarray:
        return self.pop_test.as_ndarray

    @lazy
    def y_train(self) -> np.ndarray:
        return self.pop_train.fitness.reshape(-1, 1)

    @lazy
    def y_val(self) -> np.ndarray:
        return self.pop_val.fitness.reshape(-1, 1)

    @lazy
    def y_test(self) -> np.ndarray:
        return self.pop_test.fitness.reshape(-1, 1)
