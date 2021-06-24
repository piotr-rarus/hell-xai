from typing import Tuple

import numpy as np
from evobench import Benchmark, Population
from lazy import lazy
from sklearn.pipeline import Pipeline

from .datamodule import SurrogateDataModule
from .dataset import SurrogateDataset


class SurrogateData:

    def __init__(
        self,
        benchmark: Benchmark,
        x_preprocessing: Pipeline,
        y_preprocessing: Pipeline,
        *,
        n_samples: int,
        splits: Tuple[float, float, float],
        batch_size: int,
        num_workers: int = 0
    ):
        self.benchmark = benchmark

        self.__x_preprocessing = x_preprocessing
        self.__y_preprocessing = y_preprocessing

        self.N_SAMPLES = n_samples
        self.TRAIN_SIZE = int(n_samples * splits[0])
        self.VAL_SIZE = int(n_samples * splits[1])
        self.TEST_SIZE = int(n_samples * splits[2])

        self.BATCH_SIZE = batch_size
        self.NUM_WORKERS = num_workers

    @lazy
    def data_module(self) -> SurrogateDataModule:
        return SurrogateDataModule(
            self.train_set, self.val_set, self.test_set,
            batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS
        )

    @lazy
    def train_set(self) -> SurrogateDataset:
        return SurrogateDataset(
            self.x_train, self.y_train,
            self.x_preprocessing, self.y_preprocessing
        )

    @lazy
    def val_set(self) -> SurrogateDataset:
        return SurrogateDataset(
            self.x_val, self.y_val,
            self.x_preprocessing, self.y_preprocessing
        )

    @lazy
    def test_set(self) -> SurrogateDataset:
        return SurrogateDataset(
            self.x_test, self.y_test,
            self.x_preprocessing, self.y_preprocessing
        )

    @lazy
    def x_preprocessing(self) -> Pipeline:
        self.__x_preprocessing.fit(self.x_train)
        return self.__x_preprocessing

    @lazy
    def y_preprocessing(self) -> Pipeline:
        self.__y_preprocessing.fit(self.y_train)
        return self.__y_preprocessing

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

    @lazy
    def pop_train(self) -> Population:
        pop_train = self.benchmark.initialize_population(self.TRAIN_SIZE)
        self.benchmark.evaluate_population(pop_train)

        return pop_train

    @lazy
    def pop_val(self) -> Population:
        pop_val = self.benchmark.initialize_population(self.VAL_SIZE)
        self.benchmark.evaluate_population(pop_val)

        return pop_val

    @lazy
    def pop_test(self) -> Population:
        pop_test = self.benchmark.initialize_population(self.TEST_SIZE)
        self.benchmark.evaluate_population(pop_test)

        return pop_test
