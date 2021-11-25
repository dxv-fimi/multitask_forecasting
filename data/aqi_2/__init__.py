from typing import Dict, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class AQIDataset(Dataset):
    
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index) -> Dict:
        return {
            'x': np.expand_dims(self.x[index], -1),
            'y': self.y[index]
        }

def create_data_loader(
    data_path: str,
    win_size: int,
    batch_size: int,
    num_workers: int,
    normalization_factor: float
) -> Tuple[DataLoader, DataLoader]:
    data = pd.read_csv(data_path).to_numpy(dtype=np.float32)[1:, [1, 2]] * normalization_factor
    print(data.shape)
    print(np.max(data))
    x = []
    y =[]
    for i in range(data.shape[0] - win_size):
        x.append(data[i:i+win_size])
        y.append(data[i+win_size])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    train_dataset = AQIDataset(x_train, y_train)
    test_dataset = AQIDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    return train_dataloader, test_dataloader

def test():
    create_data_loader(
        data_path='./dataset/aqi_2/data_clean_5000.csv',
        win_size=25,
        batch_size=32,
        num_workers=1,
        normalization_factor=1
    )