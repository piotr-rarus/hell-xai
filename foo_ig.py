import pandas as pd
from captum.attr import IntegratedGradients
from evobench.discrete import Trap
from evosolve.discrete import dled, lo3
import plotly.io as pio
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from hell import Surrogate, SurrogateData, plot, util
from hell.linkage_ig import EmpiricalLinkage

seed_everything(42)
pio.renderers.default = "notebook"

benchmark = Trap(blocks=[5] * 20, verbose=1)

x_preprocessing = Pipeline([
    ("standard-scaler", StandardScaler())
])

y_preprocessing = Pipeline([
    ("min-max-scaler", MinMaxScaler())
])

data = SurrogateData(
    benchmark,
    x_preprocessing, y_preprocessing,
    n_samples=1e5, splits=(0.6, 0.2, 0.2),
    batch_size=100,
)

surrogate = Surrogate(
    benchmark.genome_size,
    x_preprocessing, y_preprocessing,
    n_layers=1, learning_rate=2e-4, weight_decay=1e-8
)

early_stop_callback = EarlyStopping(
   monitor="val/r2",
   min_delta=0.000,
   patience=5,
   verbose=False,
   mode="max"
)

trainer = Trainer(
    max_epochs=2,
    gpus=1,
    progress_bar_refresh_rate=50,
    callbacks=[early_stop_callback]
)

trainer.fit(surrogate, data.data_module)
surrogate.eval()

print(benchmark.ffe)
benchmark.ffe = 0

ig_results_1 = util.test_decomposer(
    EmpiricalLinkage(
        benchmark,
        IntegratedGradients(surrogate),
        x_preprocessing,
    ),
    n_samples=100, k=10
)

# ig_results_1 = util.test_decomposer(
#     EmpiricalLinkage(
#         benchmark,
#         IntegratedGradients(surrogate),
#         x_preprocessing,
#         baseline_strategy="bounds",
#         n_baselines=10,
#         agg_fn="max",
#     ),
#     n_samples=100, k=10
# )

ig_results_zero = util.test_decomposer(
    EmpiricalLinkage(
        benchmark,
        IntegratedGradients(surrogate),
        x_preprocessing,
    ),
    n_samples=100, k=10
)

foo = 2
