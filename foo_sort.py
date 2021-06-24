%load_ext autoreload
%autoreload 2

import os
from copy import deepcopy

import pandas as pd
import plotly.express as px
import plotly.io as pio
import torch
from evobench import CompoundBenchmark, continuous, discrete
from evobench.discrete import Trap
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

from hell import Data, Surrogate
from hell.linkage import test_lo3, test_xais

import sys; sys.path.append("../../")

from captum.attr import (
    IntegratedGradients,
    NoiseTunnel,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    Saliency,
    InputXGradient,
    GuidedBackprop,
    FeatureAblation,
    FeaturePermutation,
    Occlusion,
    ShapleyValueSampling,
    KernelShap
)


seed_everything(42)
pio.renderers.default = "notebook"
