from evobench import Benchmark, Population, Solution
from pytest import fixture
from shap import Explainer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor

from hell import SurrogateData
from hell.conftest import LinkageHelpers

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
    )


@fixture(scope="module")
def explainer(data: SurrogateData) -> Explainer:

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("xgb", XGBRegressor())
        ]
    )

    # surrogate = XGBRegressor()
    pipeline.fit(data.x_train, data.y_train)

    return Explainer(pipeline[-1])


@fixture(scope="module")
def empirical_linkage(
    data: SurrogateData,
    explainer: Explainer
) -> EmpiricalLinkage:

    return EmpiricalLinkage(data.benchmark, explainer, data.x_preprocessing)


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
