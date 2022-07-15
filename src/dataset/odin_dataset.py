import os
from glob2 import glob
from typing import Callable
from src.io.io import read_asset, read_rgb
from torch.utils.data import Dataset

class ODINDataset(Dataset):
    
    def __init__(self, root: str, train: bool, transform: Callable) -> None:
        
        self.data_dir = os.path.join(root, "train" if train else "val")
        
        self.classes = [c for c in os.listdir(self.data_dir) if not c.startswith(".")]
        self.img_paths = [os.path.join(self.data_dir, c, f) for c in self.classes for f in os.listdir(os.path.join(self.data_dir, c)) if f.lower().endswith(".jpg")]
        self.asset_paths = [f.replace(".jpg", ".npy") for f in self.img_paths]
        self.labels = [int(f.split("/")[-2]) for f in self.img_paths]
        self.transform = transform
        
    def __getitem__(self, index):
        
        img_path = self.img_paths[index]
        asset_path = self.asset_paths[index]
        label = self.labels[index]
        
        img = read_rgb(img_path)
        asset = read_asset(asset_path)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, asset, label
        
        
    def __len__(self) -> int:
        return len(self.img_paths) 
         