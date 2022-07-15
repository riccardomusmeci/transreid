# import pytorch_lightning as pl
# from torch.utils.data import DataLoader
# from src.sampler.sampler import ReIDSampler
# from typing import Callable, Optional, Union, List, Dict


# # TODO: DataModule in base al dataset che gli passo (Market, ODIN) -> usa **kwargs
# class MyDataModule(pl.LightningDataModule):
    
#     def __init__(
#         self, 
#         batch_size: int,
#         dataset: Callable, # class dataset
#         sampler: Callable = None, # class sampler
#         **kwargs # --> sono quelli del dataset
#     ):
#         super().__init__()
#         self.dataset_class = dataset_class
#         self.dataset_args = **dataset_class
        
#     def prepare_data(self) -> None:
#         return super().prepare_data()
    
#     def setup(self, stage: Optional[str] = None) -> None:
#         if stage == "fit" or stage is None:
#             pass
        
#         if stage == "test" or stage is None:
#             pass
            
#     def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        
#         sampler = ReIDSampler(
#             target=self.train_dataset.labels,
#             batch_size=self.batch_size,
#             num_classes=self.num_classes,
#             num_samples=self.num_samples,
#             drop_last=self.drop_last
#         )
        
#         return DataLoader(
#             dataset=self.train_dataset,
#             batch_size=self.batch_size,
#             sampler=sampler,
#             num_workers=self.num_workers
#         )
        
#     def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        
#         return DataLoader(
#             dataset=self.val_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers
#         )
    
#     def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        
#         return DataLoader(
#             dataset=self.val_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers
#         )