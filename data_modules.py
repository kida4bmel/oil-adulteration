import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L
import pandas as pd
import numpy as np

 #Dataset Klasse
class PandasDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, feature_columns: list, label_column: str):
        self.data = dataframe
        self.features = self.data[feature_columns].values
        self.labels = self.data[label_column].values
    def __len__(self):
        return len(self.data)  
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # Für KlassifiPandasDataModule(file_path, feature_columns, label_column)zierung
        return x, y
# DataModule Klasse
class PandasDataModule(L.LightningDataModule):
    def __init__(self, file_path: str, feature_columns: list, label_column: str, batch_size: int=100, train_val_split: float = 0.8):
        super().__init__()
        self.file_path = file_path
        #self.sheet_name = sheet_name
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.dataframe = pd.read_csv(self.file_path, dtype=np.float64)
        self.dataset = PandasDataset(self.dataframe, self.feature_columns, self.label_column)
        #self.generator = generator
    def prepare_data(self):
        self.dataframe = pd.read_csv(self.file_path, dtype=np.float64) #sheet_name=self.sheet_name)  
    def setup(self, stage=None):
        self.dataset = PandasDataset(self.dataframe, self.feature_columns, self.label_column)
        train_size = int(self.train_val_split * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))  
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) 
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)  
    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)