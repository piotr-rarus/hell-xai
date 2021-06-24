from copy import deepcopy

import pandas as pd
from captum.attr import (DeepLift, DeepLiftShap, FeatureAblation,
                         FeaturePermutation, GradientShap, GuidedBackprop,
                         InputXGradient, IntegratedGradients, NoiseTunnel,
                         Saliency, ShapleyValueSampling)
from evobench.discrete import Trap
from evosolve.discrete import dled
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from hell import Surrogate, SurrogateData, plot, util
from hell.linkage import EmpiricalLinkage

seed_everything(42)

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
    # num_workers=os.cpu_count() // 2
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


xai_tests = util.test_xais(
    benchmark,
    data.x_preprocessing,
    decomposers=[
        EmpiricalLinkage(benchmark, IntegratedGradients(surrogate), x_preprocessing),
        EmpiricalLinkage(benchmark, DeepLift(surrogate), x_preprocessing),
    ],
    n_samples=100
)

dled_tests = util.test_decomposer(
    dled.EmpiricalLinkage(benchmark), n_samples=100
)

foo = 2
