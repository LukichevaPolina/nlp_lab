from torch import Tensor, nn
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import numpy as np

class CombinedDataset(Dataset):
    def __init__(self, X: csr_matrix, y: np.ndarray) -> None:
        super().__init__()
        self._X = X
        self._y = y
    
    def __len__(self):
        return len(self._y)
    
    def __getitem__(self, index):
        X, y = self._X[index], self._y[index]

        return X.toarray().reshape(-1, 14951), y