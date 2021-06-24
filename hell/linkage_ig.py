from typing import List

import numpy as np
import torch
from captum.attr import IntegratedGradients
from evobench import Benchmark, Solution
from evosolve.linkage import BaseEmpiricalLinkage, LinkageScrap
from lazy import lazy
from sklearn.pipeline import Pipeline


class EmpiricalLinkage(BaseEmpiricalLinkage):

    def __init__(
        self,
        benchmark: Benchmark,
        attr_fn: IntegratedGradients,
        x_preprocessing: Pipeline,
        baseline_strategy: str = None,  # None, random(n_samples), bounds
        n_baselines: int = None,
        agg_fn: str = "max",  # min, max, sum, mean
    ):
        super(EmpiricalLinkage, self).__init__(benchmark)
        self.attr_fn = attr_fn
        self.x_preprocessing = x_preprocessing
        self.baseline_strategy = baseline_strategy
        self.n_baselines = n_baselines
        self.agg_fn = agg_fn

        self.baselines: torch.FloatTensor = None

        if baseline_strategy == "random":
            baselines = benchmark.initialize_population(n_baselines)
            baselines = baselines.as_ndarray
            baselines = self.x_preprocessing.transform(baselines)
            self.baselines = torch.FloatTensor(baselines)

        if baseline_strategy == "bounds":
            lower_bound = self.benchmark.lower_bound.copy()
            upper_bound = self.benchmark.upper_bound.copy()
            middle = (self.benchmark.lower_bound + self.benchmark.upper_bound) / 2
            baselines = np.vstack([lower_bound, upper_bound, middle])
            baselines = x_preprocessing.transform(baselines)
            self.baselines = torch.FloatTensor(baselines)

    @lazy
    def name(self) -> str:
        return f"IG: {self.baseline_strategy} - {self.agg_fn}"

    def _get_attr(
        self,
        x: torch.FloatTensor,
        target_index: int,
    ) -> torch.FloatTensor:

        attr_base: torch.FloatTensor

        if self.baselines is None:
            x = x.reshape(1, -1)
        else:
            x = torch.vstack([x] * len(self.baselines))

        if self.baseline_strategy == "random":
            baselines = self.baselines.clone()
            baselines[:, target_index] = x[:, target_index]
            attr_base = self.attr_fn.attribute(x, self.baselines)

        elif self.baseline_strategy == "bounds":
            baselines = self.baselines.clone()
            baselines[:, target_index] = x[:, target_index]
            attr_base = self.attr_fn.attribute(x, baselines)

        else:
            # ? default, zero vector used for baseline
            attr_base = self.attr_fn.attribute(x)

        return attr_base

    def get_scrap(
        self,
        base: Solution,
        target_index: int,
        x_base: torch.FloatTensor = None,
    ) -> LinkageScrap:

        if x_base is None:
            x_base = self.x_preprocessing.transform([base.genome])
            x_base = torch.FloatTensor(x_base)
            x_base = x_base.squeeze()

        attr_base = self._get_attr(x_base, target_index)

        x_perturbed = x_base.clone()
        x_perturbed[target_index] *= -1

        attr_perturbed = self._get_attr(x_perturbed, target_index)

        if self.agg_fn == "min":
            attr_base = torch.min(attr_base, dim=0).values
            attr_perturbed = torch.min(attr_perturbed, dim=0).values
        elif self.agg_fn == "max":
            attr_base = torch.max(attr_base, dim=0).values
            attr_perturbed = torch.max(attr_perturbed, dim=0).values
        elif self.agg_fn == "mean":
            attr_base = torch.mean(attr_base, dim=0)
            attr_perturbed = torch.mean(attr_perturbed, dim=0)
        elif self.agg_fn == "sum":
            attr_base = torch.sum(attr_base, dim=0)
            attr_perturbed = torch.sum(attr_perturbed, dim=0)

        interactions = torch.abs(attr_base - attr_perturbed)
        interactions = interactions.detach().cpu().numpy().squeeze()

        return LinkageScrap(target_index, interactions)

    def get_scraps(
        self,
        bases: List[Solution],
        target_index: int,
    ) -> List[LinkageScrap]:

        scraps: List[LinkageScrap] = []

        for base in bases:
            scrap = self.get_scrap(base, target_index)
            scraps.append(scrap)

        return scraps

    def get_all_scraps(
        self,
        base: Solution,
        x_base: torch.FloatTensor = None
    ) -> List[LinkageScrap]:

        scraps: List[LinkageScrap] = []

        for target_index in range(base.genome.size):
            scrap = self.get_scrap(base, target_index, x_base)
            scraps.append(scrap)

        return scraps
