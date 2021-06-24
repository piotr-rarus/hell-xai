from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from captum.attr import (Attribution, DeepLift, DeepLiftShap, FeatureAblation,
                         GradientShap, IntegratedGradients, KernelShap,
                         ShapleyValueSampling)
from evobench import Benchmark, Solution
from evobench.linkage import metrics
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
from evosolve.linkage import BaseEmpiricalLinkage
from hell.linkage import EmpiricalLinkage

from linkage.discrete.hc import FIHC

ATTRIBUTIONS_WITH_BASELINES = [
    DeepLift,
    DeepLiftShap,
    FeatureAblation,
    GradientShap,
    IntegratedGradients,
    KernelShap,
    ShapleyValueSampling
]


def test_xais(
    benchmark: Benchmark,
    x_preprocessing: Pipeline,
    decomposers: List[EmpiricalLinkage],
    n_samples: int,
) -> pd.DataFrame:

    test_pop = benchmark.initialize_population(n_samples)
    bases = test_pop.solutions
    x_base = test_pop.as_ndarray
    x_base = x_preprocessing.transform(x_base)
    x_base = torch.FloatTensor(x_base)

    background = benchmark.initialize_population(n_samples)
    background = background.as_ndarray
    background = x_preprocessing.transform(background)
    background = torch.FloatTensor(background)

    results: List[pd.DataFrame] = []

    for decomposer in tqdm(decomposers, desc="Testing XAIs"):
        print(f"Testing: {decomposer.attribution.__class__.__name__}")
        result = test_xai(decomposer, bases, x_base, background)
        results.append(result)

    results = pd.concat(results)
    return results


def test_xai(
    decomposer: EmpiricalLinkage,
    bases: List[Solution],
    x_base: torch.Tensor,
    background: torch.Tensor,
) -> pd.DataFrame:

    hits: List[bool] = []
    mrr_scores: List[float] = []
    map_scores: List[float] = []
    ndcg_1_scores: List[float] = []
    ndcg_2_scores: List[float] = []
    ndcg_4_scores: List[float] = []

    attr_base: torch.Tensor

    use_background = type(decomposer.attribution) in ATTRIBUTIONS_WITH_BASELINES

    if use_background:
        attr_base = decomposer.attribution.attribute(x_base, background)
    else:
        attr_base = decomposer.attribution.attribute(x_base)

    for target_index in range(decomposer.benchmark.genome_size):

        scraps = decomposer.get_scraps(
            bases, target_index, x_base, attr_base,
            background if use_background else None
        )

        rankings = [scrap.ranking for scrap in scraps]
        rankings = np.vstack(rankings)
        interactions = [scrap.interactions for scrap in scraps]
        interactions = np.vstack(interactions)

        hit = interactions.sum(axis=1)
        hit = hit.astype(bool)

        mrr_score = metrics.ranking.mean_reciprocal_rank(
            target_index, rankings, decomposer.benchmark
        )
        map_score = metrics.ranking.mean_average_precision(
            target_index, rankings, decomposer.benchmark
        )
        ndcg_1_score = metrics.ranking.ndcg(
            target_index, rankings, interactions, decomposer.benchmark
        )
        ndcg_2_score = metrics.ranking.ndcg(
            target_index, rankings, interactions, decomposer.benchmark, exp_base=2
        )
        ndcg_4_score = metrics.ranking.ndcg(
            target_index, rankings, interactions, decomposer.benchmark, exp_base=4
        )

        hits += list(hit)
        mrr_scores += mrr_score
        map_scores += map_score
        ndcg_1_scores += ndcg_1_score
        ndcg_2_scores += ndcg_2_score
        ndcg_4_scores += ndcg_4_score

    results = pd.DataFrame(
        list(zip(
            [decomposer.attribution.__class__.__name__] * len(mrr_scores),
            hits, mrr_scores, map_scores,
            ndcg_1_scores, ndcg_2_scores, ndcg_4_scores
        )),
        columns=[
            "method",
            "hit", "mean_reciprocal_rank", "mean_average_precision",
            "ndcg$1", "ndcg$2", "ndcg$4"
        ]
    )

    return results


def test_decomposer(
    decomposer: BaseEmpiricalLinkage,
    n_samples: int,
    k: int
) -> pd.DataFrame:

    population = decomposer.benchmark.initialize_population(n_samples)

    hits: List[float] = []
    mrr_scores: List[float] = []
    map_scores: List[float] = []
    ndcg_1_scores: List[float] = []
    ndcg_2_scores: List[float] = []
    ndcg_4_scores: List[float] = []

    genome_size = decomposer.benchmark.genome_size

    for target_index in tqdm(range(genome_size), desc=f"Testing {decomposer.name}"):

        scraps = decomposer.get_scraps(population.solutions, target_index)

        rankings = [scrap.ranking for scrap in scraps]
        rankings = np.vstack(rankings)
        interactions = [scrap.interactions for scrap in scraps]
        interactions = np.vstack(interactions)

        hit = interactions.sum(axis=1)
        hit = hit.astype(bool)

        mrr_score = metrics.ranking.mean_reciprocal_rank(
            target_index, rankings, decomposer.benchmark, k=k
        )
        map_score = metrics.ranking.mean_average_precision(
            target_index, rankings, decomposer.benchmark, k=k
        )
        ndcg_1_score = metrics.ranking.ndcg(
            target_index, rankings, interactions, decomposer.benchmark, k=k
        )
        ndcg_2_score = metrics.ranking.ndcg(
            target_index, rankings, interactions, decomposer.benchmark, exp_base=2, k=k
        )
        ndcg_4_score = metrics.ranking.ndcg(
            target_index, rankings, interactions, decomposer.benchmark, exp_base=4, k=k
        )

        hits += list(hit)
        mrr_scores += mrr_score
        map_scores += map_score
        ndcg_1_scores += ndcg_1_score
        ndcg_2_scores += ndcg_2_score
        ndcg_4_scores += ndcg_4_score

    results = pd.DataFrame(
        list(zip(
            [f"{decomposer.name}"] * len(mrr_scores),
            hits, mrr_scores, map_scores,
            ndcg_1_scores, ndcg_2_scores, ndcg_4_scores
        )),
        columns=[
            "method",
            "hit", "mean_reciprocal_rank", "mean_average_precision",
            "ndcg$1", "ndcg$2", "ndcg$4"
        ]
    )

    return results
