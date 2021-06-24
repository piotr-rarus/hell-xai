import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


class Surrogate(LightningModule):

    def __init__(
        self,
        genome_size: int,
        x_preprocessing: Pipeline,
        y_preprocessing: Pipeline,
        n_layers: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0,
    ):
        super().__init__()
        self.save_hyperparameters(
            "genome_size",
            "n_layers",
            "learning_rate",
            "weight_decay"
        )

        self.x_preprocessing = x_preprocessing
        self.y_preprocessing = y_preprocessing

    def setup(self, stage):
        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # self.linears.append(
        #     nn.Linear(self.hparams.genome_size, self.hparams.genome_size)
        # )
        self.linears.append(
            nn.Linear(
                self.hparams.genome_size,
                self.hparams.genome_size // 2
            )
        )

        for _ in range(self.hparams.n_layers):
            in_features = self.linears[-1].out_features
            out_features = self.linears[-1].out_features // 2
            linear = nn.Linear(in_features, out_features)
            self.linears.append(linear)

        for _ in range(len(self.linears)):
            # dropout = nn.Dropout(p=0.0)
            dropout = nn.Dropout(p=0.2)
            self.dropouts.append(dropout)

        self.linear_n = nn.Linear(self.linears[-1].out_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):

        for linear, dropout in zip(self.linears, self.dropouts):
            x = linear(x)
            x = dropout(x)
            x = F.relu(x)

        x = self.linear_n(x)
        x = self.sigmoid(x)

        return x

    @torch.no_grad()
    def predict(
        self,
        x: np.ndarray,
        device: torch.device = None
    ):
        if not device:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.to(device)

        x = self.x_preprocessing.transform(x)
        x = torch.FloatTensor(x).to(device)

        y = self.forward(x)
        y = y.cpu().numpy()
        y = self.y_preprocessing.inverse_transform(y)

        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        r2 = r2_score(y.detach().cpu(), y_pred.detach().cpu())
        self.log("train/mse", loss)
        self.log("train/r2", r2)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        r2 = r2_score(y.cpu(), y_pred.cpu())
        self.log("val/mse", loss, on_step=False, on_epoch=True)
        self.log("val/r2", r2, on_step=False, on_epoch=True)        

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        r2 = r2_score(y.cpu(), y_pred.cpu())
        self.log("test/mse", loss, on_step=False, on_epoch=True)
        self.log("test/r2", r2, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train/mse",
        }
