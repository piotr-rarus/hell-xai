from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score


class Surrogate(LightningModule):

    def __init__(
        self,
        genome_size: int,
        n_layers: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0,
    ):
        super(Surrogate, self).__init__()
        self.save_hyperparameters(
            "genome_size",
            "n_layers",
            "learning_rate",
            "weight_decay"
        )

        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.linears.append(nn.Linear(genome_size, genome_size))
        # self.linears.append(nn.Linear(genome_size, genome_size // 2))

        for _ in range(n_layers):
            # linear = nn.Linear(genome_size // 2**i, genome_size // 2**(i+1))
            in_features = self.linears[-1].out_features
            out_features = self.linears[-1].out_features // 2
            linear = nn.Linear(in_features, out_features)
            self.linears.append(linear)

        for _ in range(len(self.linears)):
            # dropout = nn.Dropout(p=0.0)
            dropout = nn.Dropout(p=0.2)
            self.dropouts.append(dropout)

        self.linear_n = nn.Linear(self.linears[-1].out_features, 1)

    def forward(self, x: torch.Tensor):

        for linear, dropout in zip(self.linears, self.dropouts):
            x = linear(x)
            x = dropout(x)
            x = F.relu(x)

        x = self.linear_n(x)

        return x

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
