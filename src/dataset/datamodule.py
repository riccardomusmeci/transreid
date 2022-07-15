import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.sampler.sampler import ReIDSampler
from src.dataset.market_dataset import Market1501Dataset
from typing import Callable, Optional, Union, List, Dict

class MyDataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int, 
        num_classes: int, 
        num_samples: int,
        train_transform: Callable, 
        val_transform: Callable,
        shuffle: bool = True,
        num_workers: int = 5,
        pin_memory: bool = False,
        drop_last: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last 
        
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = Market1501Dataset(
                root=self.data_dir,
                train=True,
                transform=self.train_transform,
                pid_relabel=True
            )
            
            self.val_dataset = Market1501Dataset(
                root=self.data_dir,
                train=False,
                transform=self.val_transform,
                pid_relabel=True
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = Market1501Dataset(
                root=self.data_dir,
                train=False,
                transform=self.val_transform,
                pid_relabel=True
            )
            
    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        
        sampler = ReIDSampler(
            target=self.train_dataset.labels,
            batch_size=self.batch_size,
            num_classes=self.num_classes,
            num_samples=self.num_samples,
            drop_last=self.drop_last
        )
        
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers
        )
        
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )