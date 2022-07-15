import os
import torch
import numpy as np
from PIL import Image
from typing import List

def read_asset(asset_path: str) -> torch.Tensor:
    """Reads an asset file (npy file) and returns a tensor of assets and their occurrences

    Args:
        asset_path (str): path to asset file (npy file)

    Returns:
        torch.Tensor: asset tensor (int64)
    """
    asset = np.load(asset_path, allow_pickle=True).astype('int64')
    return torch.tensor(asset, dtype=torch.int64)

def read_rgb(file_path: str) -> np.ndarray:
    if not os.path.exists(file_path):
        raise ValueError(f"The path {file_path} does not exist")
    image = Image.open(file_path).convert("RGB")
    # TODO: remove to albumentations support
    return image
    image = np.array(image)
    return image