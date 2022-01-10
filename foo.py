
import plotly.io as pio
from evobench.continuous import cec2013lsgo

# from evosolve.continuous import dg2
from shap import Explainer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor

import hell
from hell import SurrogateData, plot, util

pio.renderers.default = "notebook"


benchmark = cec2013lsgo.F4(verbose=1)

x_preprocessing = Pipeline([
    ("standard-scaler", StandardScaler())
])

y_preprocessing = Pipeline([
    ("min-max-scaler", MinMaxScaler())
])

data = SurrogateData(
    benchmark,
    x_preprocessing, y_preprocessing,
    n_samples=2e5, splits=(0.6, 0.2, 0.2),
)

surrogate = XGBRegressor(
    n_estimators=200,
    nthread=8
)

surrogate.fit(
    data.x_train, data.y_train,
    eval_set=[(data.x_train, data.y_train), (data.x_val, data.y_val)],
    early_stopping_rounds=10, verbose=False
)

y_pred = surrogate.predict(data.x_test)
r2_score(data.y_test, y_pred)

plot.xgb_results(surrogate.evals_result())

benchmark.ffe = 0
hell_results = util.test_decomposer(
    hell.EmpiricalLinkage(
        benchmark,
        Explainer(surrogate),
        data.x_preprocessing
    ),
    n_samples=100
)
print(benchmark.ffe)


# benchmark.ffe = 0
# dg2_results = util.test_decomposer(dg2.EmpiricalLinkage(benchmark), n_samples=100)
# print(benchmark.ffe)

# %%
# results = pd.concat([hell_results, dg2_results])

# # %%
# plot.hit_ratio(results)

# # %%
# plot.ranking_metric(
#     results,
#     metric="mean_reciprocal_rank",
#     title="Mean Reciprocal Rank"
# )

# # %%
# plot.ranking_metric(
#     results,
#     metric="mean_average_precision",
#     title="Mean Average Precision"
# )

# # %%
# plot.ranking_metric(
#     results,
#     metric="ndcg",
#     title="NDCG"
# )
