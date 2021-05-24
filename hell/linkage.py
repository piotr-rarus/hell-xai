from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from captum.attr import (Attribution, DeepLift, DeepLiftShap, FeatureAblation,
                         GradientShap, IntegratedGradients, KernelShap,
                         ShapleyValueSampling)
from evobench import Benchmark
from evobench.linkage import metrics
from tqdm.auto import tqdm

from linkage.discrete import lo3
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
    attributions: List[Attribution],
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

    for attribution in tqdm(attributions, desc="Testing XAIs"):
        print(f"Testing: {attribution.__class__.__name__}")
        result = test_xai(attribution, x_base, background, benchmark)
        results.append(result)

    results = pd.concat(results)
    return results


def test_xai(
    attribution: Attribution,
    x_base: torch.Tensor,
    background: torch.Tensor,
    benchmark: Benchmark,
) -> pd.DataFrame:

    hits: List[bool] = []
    mrr_scores: List[float] = []
    map_scores: List[float] = []
    ndcg_1_scores: List[float] = []
    ndcg_2_scores: List[float] = []
    ndcg_4_scores: List[float] = []

    attr_base: torch.Tensor

    if type(attribution) in ATTRIBUTIONS_WITH_BASELINES:
        attr_base = attribution.attribute(x_base, background)
    else:
        attr_base = attribution.attribute(x_base)

    for gene_index in range(benchmark.genome_size):

        scraps, interactions = get_scraps(
            gene_index, x_base, attr_base, attribution, benchmark, background
        )

        # hit_ratio = metrics.ranking.hit_ratio(interactions)

        hit = interactions.sum(axis=1)
        hit = hit.astype(bool)
        # scraps = scraps[hits, :]
        # interactions = interactions[hits, :]

        mrr_score = metrics.ranking.mean_reciprocal_rank(gene_index, scraps, benchmark)
        map_score = metrics.ranking.mean_average_precision(
            gene_index, scraps, benchmark
        )
        ndcg_1_score = metrics.ranking.ndcg(
            gene_index, scraps, interactions, benchmark
        )
        ndcg_2_score = metrics.ranking.ndcg(
            gene_index, scraps, interactions, benchmark, exp_base=2
        )
        ndcg_4_score = metrics.ranking.ndcg(
            gene_index, scraps, interactions, benchmark, exp_base=4
        )

        hits += list(hit)
        mrr_scores += mrr_score
        map_scores += map_score
        ndcg_1_scores += ndcg_1_score
        ndcg_2_scores += ndcg_2_score
        ndcg_4_scores += ndcg_4_score

    results = pd.DataFrame(
        list(zip(
            [attribution.__class__.__name__] * len(mrr_scores),
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


def get_scraps(
    gene_index: int,
    x_base: torch.Tensor,
    attr_base: torch.Tensor,
    attribiution: Attribution,
    benchmark: Benchmark,
    background: torch.Tensor = None,
) -> Tuple[np.ndarray, np.ndarray]:

    mask = np.ones(shape=benchmark.genome_size, dtype=np.float32)
    mask[gene_index] = -1

    # mutated = torch.abs(x_base - mask)
    mutated = x_base * mask

    attr_mutated: torch.Tensor

    if type(attribiution) in ATTRIBUTIONS_WITH_BASELINES:
        attr_mutated = attribiution.attribute(mutated, background)
    else:
        attr_mutated = attribiution.attribute(mutated)

    interactions = torch.abs(attr_base - attr_mutated)
    mask = mask.astype(bool)
    mask[gene_index] = False
    interactions = interactions[:, mask]

    interactions, scraps = torch.sort(interactions, dim=1, descending=True)
    scraps[scraps >= gene_index] += 1

    interactions = interactions.detach().cpu().numpy()
    scraps = scraps.detach().cpu().numpy()

    return scraps, interactions


def test_lo3(benchmark: Benchmark, n_samples: int) -> pd.DataFrame:
    population = benchmark.initialize_population(n_samples)
    fihc = FIHC(benchmark)

    hits: List[float] = []
    mrr_scores: List[float] = []
    map_scores: List[float] = []
    ndcg_1_scores: List[float] = []
    ndcg_2_scores: List[float] = []
    ndcg_4_scores: List[float] = []

    for gene_index in tqdm(range(benchmark.genome_size), desc="Testing 3LO"):
        scraps, interactions = lo3.linkage.get_scraps_for_gene(
            gene_index, population.solutions, fihc
        )

        # hit_ratio = metrics.ranking.hit_ratio(interactions)

        hit = interactions.sum(axis=1)
        hit = hit.astype(bool)
        # scraps = scraps[hits, :]
        # interactions = interactions[hits, :]

        mrr_score = metrics.ranking.mean_reciprocal_rank(gene_index, scraps, benchmark)
        map_score = metrics.ranking.mean_average_precision(
            gene_index, scraps, benchmark
        )
        ndcg_1_score = metrics.ranking.ndcg(
            gene_index, scraps, interactions, benchmark
        )
        ndcg_2_score = metrics.ranking.ndcg(
            gene_index, scraps, interactions, benchmark, exp_base=2
        )
        ndcg_4_score = metrics.ranking.ndcg(
            gene_index, scraps, interactions, benchmark, exp_base=4
        )

        hits += list(hit)
        mrr_scores += mrr_score
        map_scores += map_score
        ndcg_1_scores += ndcg_1_score
        ndcg_2_scores += ndcg_2_score
        ndcg_4_scores += ndcg_4_score

    results = pd.DataFrame(
        list(zip(
            ["3LO"] * len(mrr_scores),
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
