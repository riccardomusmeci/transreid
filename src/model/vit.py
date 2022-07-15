import timm
import torch
import torch.nn as nn

class ViT(nn.Module):
    
    def __init__(
        self, 
        model_name: str, 
        pretrained: bool = True,
        sie_dim: int = 0,
        sie_w: float = 2.0
    ) -> None:
        """ViT initializer for embedding extraction with Side Informartion Embedding support

        Args:
            model_name (str): timm ViT model name
            pretrained (bool, optional): pretrained ViT weights. Defaults to True.
            sie_dim (int, optional): number of SIE assets. Defaults to 0.
            sie_w (float, optional): SIE weight. Defaults to 2.0.
        """
        super().__init__()
    
        self.model = timm.create_model(
            model_name=model_name, 
            pretrained=pretrained, 
            num_classes=0
        )
        self.sie_dim = sie_dim
        self.embed_dim = self.model.embed_dim
        self.sie_embed = nn.Parameter(data=torch.zeros(sie_dim, 1, self.embed_dim)) if self.sie_dim > 0 else None
        self.sie_w = sie_w
    
    def sie_val(self, side_info: torch.Tensor) -> torch.Tensor:
        """computes SIE value to for each image

        Args:
            side_info (torch.Tensor): side_info found in the image with their occurrences

        Returns:
            torch.Tensor: SIE value for each images
        """
        sie_val = torch.zeros(size=(side_info.shape[0], 1, self.embed_dim))
        for img_idx, img_side_info in enumerate(side_info):
            for asset_idx, asset_num in enumerate(img_side_info):
                sie_val[img_idx] += asset_num * self.sie_embed[asset_idx]
        return sie_val
                
    def forward(self, x: torch.Tensor, side_info: torch.Tensor) -> torch.Tensor:
        """ViT forward pass

        Args:
            x (torch.Tensor): batch images
            side_info (torch.Tensor): Side Information in the images

        Returns:
            torch.Tensor: ViT embedding
        """
                
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        
        if self.sie_embed is not None:
            sie_val = self.sie_val(side_info)
            if getattr(self.model, 'pos_embed') is not None:
                x = x + self.model.pos_embed + sie_val
            else:
                x = x + sie_val
        else:
            if getattr(self.model, 'pos_embed') is not None:
                x = x + self.model.pos_embed
        
        x = self.model.pos_drop(x)
        
        for block in self.model.blocks[:-1]:
            x = block(x)
        
        return x
        