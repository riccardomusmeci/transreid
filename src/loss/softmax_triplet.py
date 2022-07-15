import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_metric_learning.miners import BatchEasyHardMiner

class SoftmaxTripletLoss(nn.Module):
    
    def __init__(
        self,
        xent_fn = None,
        triplet_fn = None,
        class_w: float = 1.,
        triplet_w: float = 1.,
        jigsaw_w: float = 0.5,
        jigsaw: bool = True
    ) -> None:
        super().__init__()
        
        # Triplet Loss params
        self.triplet_fn = triplet_fn
        self.triplet_w = triplet_w
        self.miner = BatchEasyHardMiner()
        
        # Cross Entropy params
        self.xent_fn = xent_fn if xent_fn is not None else F.cross_entropy
        self.class_w = class_w
        
        # Jigsaw and Global branch weights
        self.jigsaw_w = jigsaw_w
        self.global_w = 1 - jigsaw_w
        
        # jigsaw
        self.jigsaw = jigsaw
    
    def _class_loss(self, scores: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """computes class loss for Jigsaw+Global branch

        Args:
            scores (torch.Tensor): scores from model
            features (torch.Tensor): features from model
            targets (torch.Tensor): targets

        Returns:
            torch.Tensor: class loss
        """
        if self.jigsaw:
            # Global Branch 
            global_class_loss = self.xent_fn(scores[:, 0, :], targets)
            # Jigsaw Branch
            _jigsaw_vals = [ self.xent_fn(scores[:, i, :], targets) for i in range(1, scores.shape[1])]
            jigsaw_class_loss = torch.mean(torch.Tensor(_jigsaw_vals))
            # Wighted Total Loss
            class_loss = self.global_w * global_class_loss + self.jigsaw_w * jigsaw_class_loss
        else:
            class_loss = self.xent_fn(scores, targets)
        
        return class_loss
        
    def _triplet_loss(self, scores: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """computes triplet loss for Jigsaw+Global branch

        Args:
            scores (torch.Tensor): scores from model
            features (torch.Tensor): features from model
            targets (torch.Tensor): targets

        Returns:
            torch.Tensor: triplet loss
        """
        if self.jigsaw:
            _triplet_vals = []
            for i in range(features.shape[1]):
                indices_tuple = self.miner(features[:, i, :], targets)
                _triplet_vals.append(self.triplet_fn(features[:, i, :], targets, indices_tuple))
            
            triplet_loss = self.global_w * _triplet_vals[0] + self.jigsaw_w * torch.mean(torch.Tensor(_triplet_vals[1:]))
        else:
            indices_tuple = self.miner(features, targets)
            triplet_loss = self.triplet_fn(features, targets, indices_tuple)
            
        return triplet_loss
            
    
    def forward(self, scores: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """computes the loss 

        Args:
            scores (torch.Tensor): model scores
            features (torch.Tensor): features from model
            targets (torch.Tensor): batch targets

        Returns:
            torch.Tensor: loss
        """
        features = torch.squeeze(features)
        scores = torch.squeeze(scores)
        targets = torch.squeeze(targets)
        
        # Class Loss (global + jigsaw branch)
        
        class_loss = self._class_loss(
            scores=scores,
            features=features,
            targets=targets
        )
        
        if self.triplet_fn is None:
            return class_loss
        
        triplet_loss = self._triplet_loss(
            scores=scores,
            features=features,
            targets=targets
        )
        
        loss = self.class_w * class_loss + self.triplet_w * triplet_loss
        return loss