import os
import re
import torch
from glob2 import glob
from src.io.io import read_rgb
from torch.utils.data import Dataset
from typing import Callable, List, Tuple

class Market1501Dataset(Dataset):
    """
    Market1501 Dataset for Person Re-Identification
    * Reference: Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    * URL: http://www.liangzheng.org/Project/project_reid.html
    Dataset statistics:
    * identities: 1501 (+1 for background)
    * images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    NUM_CAMERAS = 6
    NUM_CLASSES = 751
    def __init__(
        self, 
        root: str, 
        train: bool,
        transform: Callable = None,
        pid_begin: int = 0, # person id starting id
        pid_relabel: bool = True, # False for gallary and query
        verbose: bool = True
    ) -> None:
        super().__init__()
        
        self.pid_begin = pid_begin
        self.pid_relabel = pid_relabel
        self.transform = transform
        self.verbose = verbose
        self.train = train        
        
        if self.train:
            # if not self.pid_relabel: self.pid_relabel=True # being sure that you relabel person id in train
            train_dir = os.path.join(root, "bounding_box_train")
            self._check_before_run(train_dir)
            self.img_paths, self.labels, self.cameras = self._process_dir(train_dir)        
        else:
            # if self.pid_relabel: self.pid_relabel=False # being sure that you don't relabel person id in val/test            
            query_dir, gallery_dir = os.path.join(root, "query"), os.path.join(root, "bounding_box_test")
            query_paths, query_labels, query_cameras = self._process_dir(query_dir)
            gallery_paths, gallery_labels, gallery_cameras = self._process_dir(gallery_dir)
            self.img_paths, self.labels, self.cameras = query_paths+gallery_paths, query_labels+gallery_labels, query_cameras+gallery_cameras
            self.num_query = len(query_paths)
            
    def _check_before_run(self, data_dir):
        """Checks if all files are available before going deeper

        Args:
            data_dir (str): data dir
        Raises:
            RuntimeError: no data dir
        """
        
        if not os.path.exists(data_dir):
            raise RuntimeError("'{}' is not available".format(data_dir))
            
    def _process_dir(self, data_dir) -> Tuple[List, List, List]:
        """processes dir to get dataset sample (img_path, label, camera_id)

        Returns:
            Tuple[List, List, List]: List of image paths, List of labels, List of camera_id
        """
            
        img_paths = glob(os.path.join(data_dir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        _paths, _labels, _cameras = [], [], []
        
        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if self.pid_relabel: pid = pid2label[pid]
            
            _paths.append(img_path)
            _labels.append(self.pid_begin + pid)
            _cameras.append(camid)
            
        return _paths, _labels, _cameras
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        img_path = self.img_paths[index]
        camera = self.cameras[index]
        label = self.labels[index]
        
        img = read_rgb(img_path)
        camera = torch.zeros((self.NUM_CAMERAS), dtype=torch.int64)
        camera[self.cameras[index]] = 1
        if self.transform is not None:
            img = self.transform(img)
        
        return img, camera, label
    
    def __len__(self) -> int:
        return len(self.img_paths)
        
            
        
        
    
    