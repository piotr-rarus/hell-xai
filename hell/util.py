from typing import List

import numpy as np
import pandas as pd
from evobench.linkage import metrics
from evosolve.linkage import BaseEmpiricalLinkage
from tqdm.auto import tqdm


def test_decomposer(
    decomposer: BaseEmpiricalLinkage,
    n_samples: int,
    k: int = None
) -> pd.DataFrame:

    population = decomposer.benchmark.initialize_population(n_samples)

    hits: List[float] = []
    mrr_scores: List[float] = []
    map_scores: List[float] = []
    ndcg_scores: List[float] = []

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
        ndcg_score = metrics.ranking.ndcg(
            target_index, rankings, decomposer.benchmark, k=k
        )

        hits += list(hit)
        mrr_scores += list(mrr_score)
        map_scores += list(map_score)
        ndcg_scores += list(ndcg_score)

    results = pd.DataFrame(
        list(zip(
            [f"{decomposer.name}"] * len(mrr_scores),
            hits, mrr_scores, map_scores, ndcg_scores
        )),
        columns=[
            "method", "hit", "mean_reciprocal_rank", "mean_average_precision", "ndcg"
        ]
    )

    return results
