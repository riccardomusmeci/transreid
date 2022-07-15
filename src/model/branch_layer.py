
import torch
import torch.nn as nn
from src.model.utils import *
from typing import List, Tuple

class GlobalBranch(nn.Module):
    
    def __init__(
        self,
        trans_block: nn.Module,
        norm_layer: nn.Module,
        in_features: int,
        num_classes: int,
        bias: bool = False
    ) -> None:
        """Global Branch initializer

        Args:
            trans_block (nn.Module): transformer block
            norm_layer (nn.Module): norm layer
            in_features (int): embedding dim
            num_classes (int): num of clusters to predict
            bias (bool, optional): linear layer bias. Defaults to False.
        """
        
        super().__init__()
        
        self.trans_block = trans_block
        self.norm = norm_layer
        self.bottleneck = nn.BatchNorm1d(in_features)
        self.bottleneck.bias.requires_grad_(False)
        
        self.classifier = nn.Linear(
            in_features=in_features,
            out_features=num_classes,
            bias=bias
        )
        
        self.classifier.apply(init_weights)
        
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """GlobalLayer forward pass

        Args:
            x (torch.Tensor): embedding from ViT last layers

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: features, logits
        """
        
        x = self.trans_block(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.bottleneck(x)
        logits = self.classifier(x)
        return x, logits
         
class JigsawBranch(nn.Module):
    
    def __init__(
        self, 
        trans_block: nn.Module,
        norm_layer: nn.Module,
        in_features: int,
        num_classes: int,
        jigsaw_k: int = 4,
        shift_n: int = 5,
        shuffle_groups: int = 2,
        bias: bool = False
    ) -> None:
        """Jigsaw Branch initializer

        Args:
            trans_block (nn.Module): transformer block
            norm_layer (nn.Module): norm layer
            in_features (int): embedding dim
            num_classes (int): num of clusters to predict
            jigsaw_k (int, optional): number of jigsaw blocks. Defaults to 4.
            shuffle_groups (int, optional): number of shuffles in jigsaw branch. Defaults to 2.
            shift_n (int, optional): number of shifts in jigsaw branch. Defaults to 5.
            bias (bool, optional): linear layer bias. Defaults to False.
        """
        
        super().__init__()

        self.jigsaw_k = jigsaw_k
        self.shift_n = shift_n
        self.shuffle_groups = shuffle_groups
        
        self.trans_block = trans_block
        self.norm = norm_layer
        self.classifier_blocks = nn.ModuleList([
            nn.Linear(in_features=in_features, out_features=num_classes, bias=bias) for _ in range(self.jigsaw_k)
        ])
        for block in self.classifier_blocks:
            block.apply(init_weights)
        
        self.bottleneck_blocks = nn.ModuleList([
            nn.BatchNorm1d(in_features) for _ in range(self.jigsaw_k)
        ])
        for block in self.bottleneck_blocks:
            block.bias.requires_grad_(False)
            block.apply(init_kaiming_weights)
    
    def shift_shuffle(self, x: torch.Tensor, begin: int = 1) -> torch.Tensor:
        """Shift and shuffle operations on features

        Args:
            x (torch.Tensor): input embedding tensor
            begin (int, optional): where to begin the shift operation. Defaults to 1.

        Returns:
            torch.Tensor: shift and shuffled embeddings
        """
        batch_size, embed_dim = x.size(0), x.size(-1)
        
        # shift -> move self.shift_n initial elements and place them at the end and replace them with the ones at the end
        x = torch.cat([x[:, begin-1+self.shift_n:], x[:, begin:begin-1+self.shift_n]], dim=1)
        
        # shuffle -> with x.view() we modify the shape of the features but not the content
        try:
            x = x.view(batch_size, self.shuffle_groups, -1, embed_dim)
        except:
            x = torch.cat([x, x[:, -2:-1, :]], dim=1)
            x = x.view(batch_size, self.shuffle_groups, -1, embed_dim)
        
         # returns itself if input tensor is already contiguous, otherwise it returns a new contiguous tensor by copying data.   
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, embed_dim)
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Jigsaw forward pass

        Args:
            x (torch.Tensor): embedding from ViT last layer

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: features, logits (length of the list is jigsaw_k)
        """
        x_len = x.size(1) - 1 
        patch_len = x_len // self.jigsaw_k
        token = x[:, 0:1]
        x = self.shift_shuffle(x)
        
        features = []
        logits = []
        
        for i in range(self.jigsaw_k):
            x_patch = x[:, i*patch_len:(i+1)*patch_len]
            x_patch = torch.cat((token, x_patch), dim=1)
            x_patch = self.norm(self.trans_block(x_patch))
            x_patch = x_patch[:, 0]
            x_patch = self.bottleneck_blocks[i](x_patch)
            features.append(x_patch)
            logit = self.classifier_blocks[i](x_patch)
            logits.append(logit)
            
        return features, logits
        
        
        
    
    
        
        
        
              
        
    
    

