import copy
import torch
import torch.nn as nn
from typing import Tuple
from src.model.vit import ViT
from src.model.branch_layer import GlobalBranch, JigsawBranch

class TransReID(nn.Module):
    
    def __init__(
        self, 
        model_name: str, 
        num_classes: int,
        pretrained: bool = True,
        sie_dim: int = 0,
        sie_w: float = 2.0,
        jigsaw: bool = True,
        jigsaw_k: int = 4,
        shuffle_groups: int = 2,
        shift_n: int = 5
    ) -> None:
        """TransReID initializer

        Args:
            model_name (str): timm ViT model name
            num_classes (int): num of clusters in the dataset
            pretrained (bool, optional): ViT pretrainined weights. Defaults to True.
            sie_dim (int, optional): SIE number of assets. Defaults to 0.
            sie_w (float, optional): SIE weight. Defaults to 2.0.
            jigsaw (bool, optional): jigsaw branch enabled. Defaults to True.
            jigsaw_k (int, optional): number of jigsaw blocks. Defaults to 4.
            shuffle_groups (int, optional): number of shuffles in jigsaw branch. Defaults to 2.
            shift_n (int, optional): number of shifts in jigsaw branch. Defaults to 5.
        """
        super().__init__()
        
        # TODO: add support to other transformers backbone (BEiT, SwinT, DeiT)
        # ViT Backbone
        self.backbone = ViT(
            model_name=model_name,
            pretrained=pretrained,
            sie_dim=sie_dim,
            sie_w=sie_w
        )
        # Global Branch
        self.global_b = GlobalBranch(
            trans_block=copy.deepcopy(self.backbone.model.blocks[-1]),
            norm_layer=copy.deepcopy(self.backbone.model.norm),
            in_features=self.backbone.embed_dim,
            num_classes=num_classes
        )
        
        # Jigsaw Branch
        self.jigsaw_b = JigsawBranch(
            trans_block=copy.deepcopy(self.backbone.model.blocks[-1]),
            norm_layer=copy.deepcopy(self.backbone.model.norm),
            in_features=self.backbone.embed_dim,
            num_classes=num_classes,
            jigsaw_k=jigsaw_k,
            shift_n=shift_n,
            shuffle_groups=shuffle_groups
        ) if jigsaw else None
           
    def forward(self, x: torch.Tensor, side_info: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """TransReID forward pass

        Args:
            x (torch.Tensor): batch images
            side_info (torch.Tensor): batch Side Information

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: features, logits
        """
        x = self.backbone(x, side_info)
        global_x, global_logits = self.global_b(x)
        if self.jigsaw_b is not None:
            jigsaw_x, jigsaw_logits = self.jigsaw_b(x)
        else:
            jigsaw_x, jigsaw_logits = [], []
        
        return (
            torch.stack([global_x] + jigsaw_x, dim=1),
            torch.stack([global_logits] + jigsaw_logits, dim=1)
        )
