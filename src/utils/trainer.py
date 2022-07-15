import os
from typing import List
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, LightningLoggerBase

def get_callbacks(output_dir: str) -> List[pytorch_lightning.Callback]:
    """Returns list of pl.Callbacks

    Args:
        output_dir (str): output dir

    Returns:
        List[pytorch_lightning.Callback]: list of pl.Callbacks
    """
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoint"),
            filename="epoch={epoch}-step={step}-val_loss={loss/val:.3f}-val_cm@1c={cmc@1/val:.3f}-val_mAP={mAP/val:.3f}",
            monitor="cmc@1/val",
            verbose=True,
            mode="max",
            save_top_k=5,
            auto_insert_metric_name=False
        )
    )
    
    callbacks.append(
         LearningRateMonitor(
             logging_interval="epoch",
             log_momentum=True
        )
    )
    
    return callbacks

def get_logger(output_dir: str) -> LightningLoggerBase:
    """returns a logger

    Args:
        output_dir (str): output dir
        name (str): name
        version (str): versione

    Returns:
        pytorch_lightning.loggers.LightningLoggerBase: logger
    """
    return TensorBoardLogger(
        save_dir=os.path.join(output_dir, "tensorboard")
    )