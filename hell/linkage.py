import torch
from captum.attr import Attribution
from evobench import Benchmark, Solution
from evosolve.linkage import BaseEmpiricalLinkage, LinkageScrap
from sklearn.pipeline import Pipeline
from typing import List


class EmpiricalLinkage(BaseEmpiricalLinkage):

    def __init__(
        self,
        benchmark: Benchmark,
        attribution: Attribution,
        x_preprocessing: Pipeline
    ):
        super(EmpiricalLinkage, self).__init__(benchmark)
        self.attribution = attribution
        self.x_preprocessing = x_preprocessing

    def get_scrap(
        self,
        base: Solution,
        target_index: int,
        x_base: torch.FloatTensor = None,
        attr_base: torch.FloatTensor = None,
        background: torch.FloatTensor = None
    ) -> LinkageScrap:

        # if x_base is None:
        #     x_base = self.x_preprocessing.transform([base.genome])
        #     x_base = torch.FloatTensor(x_base)

        # if attr_base is None:
        #     # if background is not None:
        #     #     attr_base = self.attribution.attribute(x_base, background)
        #     # else:
        #     #     attr_base = self.attribution.attribute(x_base)

        #     attr_base = self.attribution.attribute(x_base)

        perturbed = x_base.clone()
        perturbed[target_index] *= -1
        perturbed = perturbed.reshape(1, -1)

        attr_perturbed: torch.Tensor

        if background is not None:
            background = background.reshape(1, -1)
            attr_perturbed = self.attribution.attribute(perturbed, background)
        else:
            attr_perturbed = self.attribution.attribute(perturbed)

        attr_perturbed = attr_perturbed.squeeze()

        interactions = torch.abs(attr_base - attr_perturbed)
        interactions = interactions.detach().cpu().numpy().squeeze()

        return LinkageScrap(target_index, interactions)

    def get_scraps(
        self,
        bases: List[Solution],
        target_index: int,
        x_base: torch.FloatTensor = None,
        attr_base: torch.FloatTensor = None,
        background: torch.FloatTensor = None
    ) -> List[LinkageScrap]:

        scraps: List[LinkageScrap] = []

        if background is not None:
            for base, x_b, attr_b, baseline in zip(bases, x_base, attr_base, background):
                scrap = self.get_scrap(base, target_index, x_b, attr_b, baseline)
                scraps.append(scrap)
        else:
            for base, x_b, attr_b in zip(bases, x_base, attr_base):
                scrap = self.get_scrap(base, target_index, x_b, attr_b)
                scraps.append(scrap)

        return scraps

    def get_all_scraps(
        self,
        base: Solution,
        x_base: torch.FloatTensor = None,
        attr_base: torch.FloatTensor = None,
        background: torch.FloatTensor = None
    ) -> List[LinkageScrap]:

        scraps: List[LinkageScrap] = []

        for target_index in range(base.genome.size):
            scrap = self.get_scrap(base, target_index, x_base, attr_base, background)
            scraps.append(scrap)

        return scraps
