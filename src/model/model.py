import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from src.model.transreid import TransReID
from src.metric.retrieval import CmC_mAP
from src.loss.softmax_triplet import SoftmaxTripletLoss
from pytorch_metric_learning.losses import TripletMarginLoss


class ReIDModel(pl.LightningModule):
    
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
        shift_n: int = 5,
    ) -> None:
        
        super().__init__()
        self.model = TransReID(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            sie_dim=sie_dim,
            sie_w=sie_w,
            jigsaw=jigsaw,
            jigsaw_k=jigsaw_k,
            shuffle_groups=shuffle_groups,
            shift_n=shift_n
        )
        
        self.loss = SoftmaxTripletLoss(
            xent_fn=torch.nn.CrossEntropyLoss(),
            triplet_fn=TripletMarginLoss(),
            jigsaw=jigsaw
        )
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.4)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10, 
            T_mult=2, 
            eta_min=0
        )
        self.accuracy = Accuracy()
        self.cmc_map = CmC_mAP()
        
    def forward(self, x, asset):
        return self.model(x, asset)
    
    def training_step(self, batch, batch_idx):
        
        x, asset, target = batch
        features, logits = self(x, asset)
        # loss = self.loss(logits.squeeze(), target)
        loss = self.loss(logits, features, target)
        # accuracy update
        preds = logits[:, 0].max(1).indices
        self.accuracy.update(
            preds=preds, 
            target=target
        )
        self.log("loss/train", loss, sync_dist=True, prog_bar=True)
        self.log("accuracy/train", self.accuracy.compute(), prog_bar=True, sync_dist=True,)
        self.log("lr", self.lr_scheduler.get_last_lr()[0], prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x, asset, target = batch
        features, logits = self(x, asset)
        
        # loss = self.loss(logits.squeeze(), target)
        loss = self.loss(logits, features, target)
        
        self.cmc_map.update(
            features=features, 
            target=target
        )
        cmc_vals, mAP = self.cmc_map.compute()
        
        self.log("loss/val", loss, sync_dist=True, prog_bar=True)
        self.log("cmc@1/val", cmc_vals[0], prog_bar=True, sync_dist=True)
        self.log("mAP/val", mAP, prog_bar=True, sync_dist=True)
    
    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return [self.optimizer]
        else:
            return [self.optimizer], [self.lr_scheduler]