import torch
import numpy as np
from typing import List

def euclidean_distance(x: torch.Tensor, y: torch.Tensor, fill_diag: bool = True) -> np.array:
    """Computes the Euclidean Distance Matrix between features

    Args:
        x (torch.Tensor): first torch.Tensor
        y (torch.Tensor): second torch.Tensor
        fill_diag (bool, optiona): whether to fill diagonal with huge value (only if matrix is squared). Defaults to True.

    Returns:
        np.array: matrix containing distances between features
    """
    m = x.shape[0]
    n = y.shape[0]
    
    _dmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    _dmat.addmm_(
        mat1=x,
        mat2=y.t(),
        beta=1,
        alpha=-2
    )
    _dmat = _dmat.cpu().detach().numpy()
    if fill_diag:
        n = _dmat.shape[0]
        _dmat[range(n), range(n)] = np.max(_dmat)*10
    
    return _dmat