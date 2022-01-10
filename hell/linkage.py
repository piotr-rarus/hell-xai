from typing import List

import numpy as np
from evobench import Benchmark, Solution
from evosolve.linkage import BaseEmpiricalLinkage, LinkageScrap
from shap import Explainer
from sklearn.pipeline import Pipeline


class EmpiricalLinkage(BaseEmpiricalLinkage):

    def __init__(
        self,
        benchmark: Benchmark,
        explainer: Explainer,
        x_preprocessing: Pipeline
    ):
        super(EmpiricalLinkage, self).__init__(benchmark)
        self.explainer = explainer
        self.x_preprocessing = x_preprocessing

    def get_scrap(
        self,
        base: Solution,
        target_index: int,
        x_base: np.ndarray = None,
        attr_base: np.ndarray = None,
    ) -> LinkageScrap:

        if x_base is None:
            x_base = self.x_preprocessing.transform([base.genome])

        if attr_base is None:
            attr_base = self.explainer(x_base).values.squeeze()

        perturbed = x_base.copy()
        perturbed[target_index] *= -1

        attr_perturbed = self.explainer(perturbed).values.squeeze()
        interactions = np.abs(attr_base - attr_perturbed)

        return LinkageScrap(target_index, interactions)

    def get_scraps(
        self,
        bases: List[Solution],
        target_index: int,
        x_base: np.ndarray = None,
        attr_base: np.ndarray = None
    ) -> List[LinkageScrap]:

        scraps: List[LinkageScrap] = []

        if x_base is None:
            x_base = [base.genome for base in bases]
            x_base = self.x_preprocessing.transform(x_base)

        if attr_base is None:
            attr_base = self.explainer(x_base).values

        perturbed = x_base.copy()
        perturbed[:, target_index] *= -1

        attr_perturbed = self.explainer(perturbed).values
        interactions = np.abs(attr_base - attr_perturbed)

        scraps = [
            LinkageScrap(target_index, solution_interactions)
            for solution_interactions in interactions
        ]

        return scraps

    def get_all_scraps(
        self,
        base: Solution,
        x_base: np.ndarray = None,
        attr_base: np.ndarray = None,
        background: np.ndarray = None
    ) -> List[LinkageScrap]:

        scraps: List[LinkageScrap] = []

        for target_index in range(base.genome.size):
            scrap = self.get_scrap(base, target_index, x_base, attr_base, background)
            scraps.append(scrap)

        return scraps
