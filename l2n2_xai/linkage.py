from typing import List

import numpy as np
import pandas as pd
import torch
from captum.attr import (Attribution, DeepLift, DeepLiftShap, FeatureAblation,
                         GradientShap, IntegratedGradients, KernelShap,
                         ShapleyValueSampling)
from evobench import Benchmark
from sklearn.metrics import ndcg_score
from tqdm.auto import tqdm

XAI_WITH_BASELINES = [
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
    xais: List[Attribution],
    n_samples: int
) -> pd.DataFrame:

    test_pop = benchmark.initialize_population(n_samples)
    x_base = test_pop.as_ndarray.astype(np.float32)
    x_base = torch.FloatTensor(x_base)

    background = benchmark.initialize_population(n_samples)
    background = background.as_ndarray.astype(np.float32)
    background = torch.FloatTensor(background)

    x_base *= 2
    background *= 2

    x_base -= 1
    background -= 1

    results: List[pd.DataFrame] = []

    for xai in tqdm(xais, desc="Testing XAIs"):
        print(f"Testing: {xai.__class__.__name__}")

        result = test_xai(benchmark, xai, x_base, background)
        results.append(result)

    results = pd.concat(results)
    return results


def test_xai(
    benchmark: Benchmark,
    xai: Attribution,
    x_base: torch.Tensor,
    background: torch.Tensor,
) -> pd.DataFrame:

    mrr_scores: List[float] = []
    map_scores: List[float] = []
    ndcg_scores: List[float] = []

    attr_base: torch.Tensor

    if type(xai) in XAI_WITH_BASELINES:
        attr_base = xai.attribute(x_base, background)
    else:
        attr_base = xai.attribute(x_base)

    for gene_index in range(benchmark.genome_size):
        mask = np.ones(shape=benchmark.genome_size, dtype=np.float32)
        mask[gene_index] = -1

        # mutated = torch.abs(x_base - mask)
        mutated = x_base * mask

        attr_mutated: torch.Tensor

        if type(xai) in XAI_WITH_BASELINES:
            attr_mutated = xai.attribute(mutated, background)
        else:
            attr_mutated = xai.attribute(mutated)

        interactions = torch.abs(attr_base - attr_mutated)

        mask[gene_index] = 0
        mask = mask.astype(bool)
        true_linkage = benchmark.true_dsm[gene_index][mask].astype(bool)
        interactions = interactions[:, mask]

        gene_ranking = torch.argsort(interactions, dim=1, descending=True)
        gene_ranking = gene_ranking.cpu().numpy()
        relevance = true_linkage[gene_ranking]

        linkage, _ = torch.sort(interactions, dim=1, descending=True)
        linkage = linkage.detach().cpu().numpy()

        mrr_score = [mean_reciprocal_rank(ranking) for ranking in relevance]
        map_score = [mean_average_precision(ranking) for ranking in relevance]

        mrr_scores += mrr_score
        map_scores += map_score

        relevance = true_linkage[gene_ranking].astype(float)

        for scrap, true_scores in zip(linkage, relevance):
            score = ndcg_score([scrap], [true_scores])
            ndcg_scores.append(score)

    results = pd.DataFrame(
        list(zip(
            [xai.__class__.__name__] * len(mrr_scores),
            mrr_scores,
            map_scores,
            ndcg_scores
        )),
        columns=["xai", "mean_reciprocal_rank", "mean_average_precision", "ndcg"]
    )

    return results


def mean_reciprocal_rank(ranking: List[bool]) -> float:
    ranks = np.nonzero(ranking)[0]
    min_index = ranks.min() if ranks.size else float("inf")
    rank = 1 / (min_index + 1)
    return rank


def mean_average_precision(ranking: List[bool]) -> float:
    ranks = np.nonzero(ranking)[0]

    scores = []

    for index, rank in enumerate(ranks):
        average_precision = (index + 1) / (rank + 1)
        scores.append(average_precision)

    score = 0

    if scores:
        score = np.mean(scores)

    return score
