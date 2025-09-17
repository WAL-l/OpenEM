import os
import random

import numpy as np
from torch.utils.data import DataLoader, Dataset
import lightning as L


class NoiseDataset(Dataset):
    def __init__(
            self,
            data_paths,
    ):
        super().__init__()
        self.root_dir = data_paths
        self.data_files = []

        for subdir in os.listdir(data_paths):
            subdir_path = os.path.join(data_paths, subdir)
            if os.path.isdir(subdir_path):
                model_dir = os.path.join(subdir_path, 'model')
                self.data_files.extend(
                    [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        model_path = self.data_files[idx]
        data_path = model_path.replace('model', 'data')
        height_path = model_path.replace('model', 'height')

        model = np.load(model_path)
        model = np.log(model)
        model = model / 3

        data = np.load(data_path)
        data = np.log(np.abs(data))
        data_min = data.min()
        data_max = data.max()
        data = (data - data_min) * 2 / (data_max - data_min) - 1

        model = model[np.newaxis, :]
        data = data[np.newaxis, :]

        model = (model.astype(np.float32))
        data = (data.astype(np.float32))

        height_data = np.load(height_path)
        height = height_data.astype(np.float32)

        return {'model': model, 'data': data, 'height': height}


class DataModule(L.LightningDataModule):
    def __init__(self, train_dir: str = "./test_data", val_dir: str = "./test_data", batch_size: int = 1,
                 num_workers: int = 0):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_set = NoiseDataset(data_paths=self.train_dir)
        self.val_set = NoiseDataset(data_paths=self.val_dir)

    def train_dataloader(self):
        ld_train = DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )
        return ld_train

    def val_dataloader(self):
        ld_val = DataLoader(
            self.val_set,
            num_workers=0,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        return ld_val


if __name__ == '__main__':
    root_dir = './dataset/train'
    dataset = NoiseDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 遍历数据加载器
    for batch in dataloader:
        # 在这里进行训练或其他操作
        label = batch['data'][0][0].cpu().numpy()

        print(batch.shape, batch.shape)
