from torch.utils.data import DataLoader
from src.metric.retrieval import CmC_mAP
from src.model.transreid import TransReID
from src.dataset.odin_dataset import ODINDataset
from src.dataset.market_dataset import Market1501Dataset
from src.transform.transform import get_transforms
from tqdm import tqdm
import torch

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


market_data_dir = "/Users/riccardomusmeci/Developer/enel/data/transreid/dataset/Market-1501-v15.09.15"
market_query_dir = "/Users/riccardomusmeci/Developer/enel/data/transreid/dataset/Market-1501-v15.09.15/query"
market_gallery_dir = "/Users/riccardomusmeci/Developer/enel/data/transreid/dataset/Market-1501-v15.09.15/bounding_box_test"

odin_data_dir = "/Users/riccardomusmeci/Developer/enel/data/transreid/dataset/ODIN"

MARKET_CLASSES = 751
MARKET_SIE_DIM = 6

# val_dataset = ODINDataset(
#     root=odin_data_dir,
#     train=False,
#     transform=get_transforms('val', img_size=(224, 224))
# )

# train_dataset = Market1501Dataset(
#     root_dir=market_data_dir,
#     train=True,
#     pid_relabel=True,
#     transform=get_transforms('train', img_size=(224, 224))
# )

val_dataset = Market1501Dataset(
    root_dir=market_data_dir,
    train=False,
    pid_relabel=False,
    transform=get_transforms('val', img_size=(224, 224))
)

data_loader = DataLoader(
    dataset=val_dataset,
    batch_size=32
)

print(Market1501Dataset.NUM_CAMERAS)
print(Market1501Dataset.NUM_CLASSES)

model = TransReID(
    model_name="vit_large_patch16_224",
    pretrained=False,
    num_classes=Market1501Dataset.NUM_CLASSES,
    sie_dim=Market1501Dataset.NUM_CAMERAS
)

cmc_map_eval = CmC_mAP(num_query=None)
model.eval()
for batch in tqdm(data_loader):
    with torch.no_grad():
        x, sie, target = batch
        features, logits = model(x, sie)
        cmc_map_eval.update(
            features=features, 
            target=target
        )
cmc_vals, mAP = cmc_map_eval.compute()
print(f"cmc@1 {cmc_vals[0]} - cmc@3 {cmc_vals[2]} - cmc@5 {cmc_vals[4]}")
print(f"mAP {mAP}")

    
