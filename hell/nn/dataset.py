import numpy as np
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset


class SurrogateDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray, y: np.ndarray,
        x_preprocessing: Pipeline,
        y_preprocessing: Pipeline,
    ):
        super().__init__()
        self.x = x_preprocessing.transform(x).astype(np.float32)
        self.y = y_preprocessing.transform(y).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
