from src.data.components.combined_dataset import CombinedDataset

from torch.utils.data import DataLoader

class CombinedDatamodule:
    def __init__(self, X_train, y_train, X_test, y_test, train_bs, test_bs) -> None:
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._train_bs = train_bs
        self._test_bs = test_bs
    
    def setup(self):
        self._train_dataset = CombinedDataset(X=self._X_train, y=self._y_train)
        self._test_dataset = CombinedDataset(X=self._X_test, y=self._y_test)

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self._train_bs, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataloader, batch_size=self._test_bs, shuffle=False)