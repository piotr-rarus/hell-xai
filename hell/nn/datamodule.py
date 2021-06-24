from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class SurrogateDataModule(LightningDataModule):
    def __init__(
        self,
        train_set: Dataset, val_set: Dataset, test_set: Dataset,
        batch_size: int, num_workers: int
    ):
        super().__init__()

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.BATCH_SIZE = batch_size
        self.NUM_WORKERS = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            shuffle=False
        )
