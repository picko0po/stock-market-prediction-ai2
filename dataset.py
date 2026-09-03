import torch
from torch.utils.data import Dataset


class LSTMStocksDataset(Dataset):

    def init(self, x_tensor, y_tensor):
        self._x_tensor = x_tensor.float()
        self._y_tensor = y_tensor.float()

    def len(self):
        return len(self._y_tensor)

    def getitem(self, idx):
        return  self._x_tensor[idx],  self._y_tensor[idx]
