import torch
from captum.attr import Attribution, IntegratedGradients, Saliency
from evobench import Benchmark, Population, Solution
from hell import Surrogate, SurrogateData
from hell.conftest import LinkageHelpers
from pytest import fixture
from pytorch_lightning import Trainer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..linkage import EmpiricalLinkage


@fixture(scope="module")
def data(benchmark: Benchmark) -> SurrogateData:
    x_preprocessing = Pipeline([
        ("standard-scaler", StandardScaler())
    ])

    y_preprocessing = Pipeline([
        ("min-max-scaler", MinMaxScaler())
    ])

    return SurrogateData(
        benchmark,
        x_preprocessing, y_preprocessing,
        n_samples=100, splits=(0.6, 0.2, 0.2),
        batch_size=10,
    )


@fixture(scope="module")
def attribution(data: SurrogateData) -> Attribution:

    surrogate = Surrogate(
        data.benchmark.genome_size,
        data.x_preprocessing, data.y_preprocessing,
        n_layers=0, learning_rate=1e-3, weight_decay=1e-6
    )

    trainer = Trainer(
        max_epochs=5,
        gpus=1,
        progress_bar_refresh_rate=50,
    )

    trainer.fit(surrogate, data.data_module)
    surrogate.eval()

    return Saliency(surrogate)


@fixture(scope="module")
def empirical_linkage(
    data: SurrogateData,
    attribution: Attribution
) -> EmpiricalLinkage:

    return EmpiricalLinkage(data.benchmark, attribution, data.x_preprocessing)


def test_get_scrap(
    empirical_linkage: EmpiricalLinkage,
    solution: Solution,
    linkage_helpers: LinkageHelpers
):
    scrap = empirical_linkage.get_scrap(solution, target_index=0)
    genome_size = empirical_linkage.benchmark.genome_size

    linkage_helpers.check_empirical_scraps([scrap], genome_size)


def test_get_scraps(
    empirical_linkage: EmpiricalLinkage,
    population: Population,
    linkage_helpers: LinkageHelpers
):
    scraps = empirical_linkage.get_scraps(population.solutions, target_index=0)
    genome_size = empirical_linkage.benchmark.genome_size

    linkage_helpers.check_empirical_scraps(scraps, genome_size)


def test_get_all_scraps(
    empirical_linkage: EmpiricalLinkage,
    solution: Solution,
    linkage_helpers: LinkageHelpers
):
    scraps = empirical_linkage.get_all_scraps(solution)
    genome_size = empirical_linkage.benchmark.genome_size

    linkage_helpers.check_empirical_scraps(scraps, genome_size)
