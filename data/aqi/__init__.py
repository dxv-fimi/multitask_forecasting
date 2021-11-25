from typing import Dict

import numpy as np
import pandas as pd
from torch.utils import data
from torch.utils.data import DataLoader, Dataset


class AQI(Dataset):

    def __init__(self, data_path, win_size, normalize_factor) -> None:
        super().__init__()
        self.data_path = data_path
        self.data = pd.read_csv(data_path).to_numpy(dtype=np.float32) * normalize_factor
        self.win_size = win_size

    def __len__(self):
        return self.data.shape[0] - self.win_size + 1

    def __getitem__(self, index: int) -> Dict:
        return {
            'x': np.expand_dims(self.data[index:index+self.win_size-1], -1),
            'y': self.data[index+self.win_size-1]
        }

def create_data_loader(data_path, win_size, batch_size, num_workers, normalize_factor):
    dataset = AQI(data_path, win_size, normalize_factor)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
