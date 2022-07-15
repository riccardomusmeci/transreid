import torch
import numpy as np
from typing import List, Tuple
from src.metric.utils import euclidean_distance

# TODO: implement as torchmetrics.Metric class -> https://torchmetrics.readthedocs.io/en/stable/pages/implement.html
class CmC_mAP():

    def __init__(self, num_query: int = None, max_rank=30, features_norm: bool = True) -> None:
        """CmC and mAP evaluator

        Args:
            num_query (int, optional): number of query in the dataset. Defaults to None.
            max_rank (int, optional): max rank. Defaults to 30.
            features_norm (bool, optional): feature normalization. Defaults to True.
        """
        # Number of images in the validation set
        self.num_query = num_query
        self.max_rank = max_rank
        self.features_norm = features_norm
        self.features = []
        self.targets = []
        
    def reset(self):
        self.features = []
        self.targets = []

    def update(self, features: List[torch.Tensor], target: torch.Tensor):
        """Updates features and target (must be called for each batch of images)

        Args:
            features (List[torch.Tensor]): list of torch tensors containing batch features
            targets (torch.Tensor): target for each set of feature in the batch
        """
        N, B, F = features.shape
        features = features.view((N, B*F))
        self.features.append(features)
        self.targets.extend(np.asarray(target))

    def eval_fn(self, dmat: np.array, query_target: np.array, gallery_target: np.array, max_rank=50):
        
        num_query, num_gallery = dmat.shape
        if num_gallery < max_rank:
            print(f"Note: number of gallery samples is quite small, got {num_gallery} while max_rank is set to {max_rank}.")
            max_rank = num_gallery

        max_rank = num_gallery
        
        # Sorting for each row from lower dist to higher dist (based on query features)
        indices = np.argsort(dmat, axis=1)
        
        # Finding match between
        matches = (gallery_target[indices] == query_target[:, np.newaxis]).astype(np.int32)
        
        all_cmc = []
        all_AP = []
        num_valid_q = 0
        for q_idx in range(num_query):
            # getting matches for a query
            orig_cmc = matches[q_idx]
            # if orig_cmc is [1, 0, 0, 1], orig_cmc.cumsum() is [1, 1, 1, 2]
            cmc = orig_cmc.cumsum()
            cmc[cmc>1] = 1
            all_cmc.append(cmc[:max_rank])
        
            num_valid_q += 1.

            # Computing average precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
            tmp_cmc = tmp_cmc / y
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            ap = tmp_cmc.sum() / num_rel
            all_AP.append(ap)

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evals the model after each epoch
        """
        # Concatenating features in a single torch.Tensor
        features = torch.cat(self.features, dim=0)

        if self.features_norm is True:
            # print("Features in Evaluator are normalized")
            features = torch.nn.functional.normalize(features, dim=1, p=2)

        # Getting query features and target
        # TODO: test con diverse query e diverse gallery
        query_features = features if self.num_query is None else features[:self.num_query]
        query_target = np.asarray(self.targets) if self.num_query is None else np.asarray(self.targets[:self.num_query])
        # Getting gallery features and target
        gallery_features = features if self.num_query is None else features[self.num_query:]
        gallery_target = np.asarray(self.targets) if self.num_query is None else np.asarray(self.targets[self.num_query:])
        
        # print("Computing distance matrix with euclidean distance")
        dmat = euclidean_distance(
            x=query_features,
            y=gallery_features,
            fill_diag=True
        )

        cmc, mAP = self.eval_fn(
            dmat=dmat,
            query_target=query_target,
            gallery_target=gallery_target,
            max_rank=self.max_rank
        )

        return cmc, mAP
        
        