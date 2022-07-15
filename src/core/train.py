import os
import pytorch_lightning as pl
from src.model.model import ReIDModel
from src.dataset.datamodule import MyDataModule
from src.transform.transform import get_transforms
from src.utils.trainer import get_callbacks, get_logger
from src.dataset.market_dataset import Market1501Dataset

def train():
    
    pl.seed_everything(seed=42, workers=True)
    # data_dir = "/Users/riccardomusmeci/Developer/enel/data/transreid/dataset/ODIN"
    data_dir = "/opt/ml/input/data/dataset"
    output_dir = "/opt/ml/output"
    
    print(os.listdir(data_dir))

    num_classes = Market1501Dataset.NUM_CLASSES
    sie_dim = Market1501Dataset.NUM_CAMERAS
    batch_size = 64
    num_samples = 4
    
    datamodule = MyDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_classes=num_classes,
        num_samples=num_samples,
        train_transform=get_transforms('train'),
        val_transform=get_transforms('val')
    )
    
    model = ReIDModel(
        model_name="vit_small_patch16_224",
        num_classes=num_classes,
        sie_dim=sie_dim,
        jigsaw=True
    )
    
    logger = get_logger(output_dir=output_dir)
    callbacks = get_callbacks(output_dir=output_dir)
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=output_dir,
        gpus=1,
        max_epochs=100,
        precision=16,
        log_gpu_memory=True, 
    )
    
    trainer.fit(
        model=model,
        datamodule=datamodule
    )