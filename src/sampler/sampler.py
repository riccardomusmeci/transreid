import copy
import enum
import random
import numpy as np
from typing import Iterable, List
from torch.utils.data import Sampler
from src.sampler.utils import fill_random

class ReIDSampler(Sampler):
    
    def __init__(
        self, 
        target: List[int], 
        batch_size: int, 
        num_classes: int,
        num_samples: int = 2,
        drop_last: bool = False
    ) -> None:
        """ReIdentification Sampler that returns a number of fixed samples in a batch for a class

        Args:
            target (List[int]): images targets
            batch_size (int): batch size
            num_classes (int): number of classes in the dataset
            num_samples (int, optional): number of samples per class in a batch. Defaults to 4.
            drop_last (bool, optional): set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If False and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. Defaults to False.
        """
        
        self.target = target
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_samples_class = num_samples
        self.num_samples_batch = self.batch_size // self.num_samples_class
        self.drop_last = drop_last
    
        self._indices = None
        self._length = None
    
    @property
    def indices(self):
        if self._indices is not None:
            return self._indices
        else:
            self._indices = [[] for _ in range(self.num_classes)]
            for i, t in enumerate(self.target):
                self._indices[t].append(i)
            return self._indices
                
    @property
    def length(self):
        
        if self._length is not None:
            return self._length
        else:
            self._length = 0
            for target_indices in self.indices:
                # if number of samples for a target class is less that self.num_samples, than assing self.num_samples
                num_samples = self.num_samples_class if len(target_indices) < self.num_samples_class else len(target_indices)
            self._length += num_samples - num_samples % self.num_samples_class
    
    def __iter__(self) -> Iterable:
        
        batch_indices = [[] for _ in range(self.num_classes)]
        classes = [i for i in range(0, self.num_classes)]
        
        for c in classes:
            class_indices = self.indices[c]
            # if we have less samples than minimum num_samples, we randomly increase the samples up to self.num_samples
            if len(class_indices) < self.num_samples_class and len(class_indices) > 0:
                class_indices = fill_random(
                    l=class_indices,
                    size=self.num_samples_batch
                )
            # random shuffle of the _indices
            random.shuffle(class_indices)
            # creating indices of size self.num_samples_class for each class (e.g. class 0 -> [[1, 10, 3, 5], [5, 2, 1, 10], ..])
            _indices = []
            for i in class_indices:
                _indices.append(i)
                if len(_indices) == self.num_samples_class:
                    batch_indices[c].append(_indices)
                    _indices = []
            
        indices = []
        classes_with_indices = [i for i in classes if len(batch_indices[i]) > 0]
        while True:
            # randomnly selecting classes for a batch
            classes_with_indices = [i for i in classes if len(batch_indices[i]) > 0]            
            if len(classes_with_indices) > self.num_samples_batch:
                sel_classes = random.sample(
                    classes_with_indices,
                    k=self.num_samples_batch
                )
                for c in sel_classes:
                    _indices = batch_indices[c].pop(0) # set of indices for class c
                    indices.extend(_indices)
            else:
                if not self.drop_last:
                    sel_classes = classes_with_indices
                    _indices = [el for c in sel_classes for el in batch_indices[c].pop(0)]
                    _indices = fill_random(
                        l=_indices,
                        size=self.batch_size
                    )
                    indices.extend(_indices)
                break
        return iter(indices)
        
    def __len__(self) -> int:
        return self._length
            
        
                
            
            
