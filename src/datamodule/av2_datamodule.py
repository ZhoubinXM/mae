from pathlib import Path
from typing import Optional
from sklearn.cluster import KMeans
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count

from tqdm import tqdm

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader

from .av2_dataset import Av2Dataset, collate_fn, sept_collate_fn, sept_collate_fn_lane_candidate


class Av2DataModule(LightningDataModule):

    def __init__(
        self,
        data_root: str,
        data_folder: str,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        test: bool = False,
    ):
        super(Av2DataModule, self).__init__()
        self.data_root = Path(data_root)
        self.data_folder = data_folder
        if data_folder in ["model_mae_sept"]:
            self.data_folder = "forecast-mae"
        if data_folder in ["model_sept", "model_sept_mae"]:
            self.data_folder = "forecast-sept-dev"
            # purn method
            # self.data_folder = "model-sept-all"
            self.collate_fn = sept_collate_fn
        else:
            self.collate_fn = collate_fn
        self.batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test = test

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            self.train_dataset = Av2Dataset(data_root=self.data_root /
                                            self.data_folder,
                                            cached_split="train")
            self.val_dataset = Av2Dataset(data_root=self.data_root /
                                          self.data_folder,
                                          cached_split="val")
        else:
            self.test_dataset = Av2Dataset(data_root=self.data_root /
                                           self.data_folder,
                                           cached_split="test")

    def train_dataloader(self):
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )


if __name__ == "__main__":
    TYPE_MAP = {0: "Vehicle", 1: "Pedestrain", 2: "Cyclist"}
    data_root = "/data/jerome.zhou/prediction_dataset/av2/"
    anchor_save_path = os.path.join(data_root, 'anchor')
    if not os.path.exists(anchor_save_path):
        os.makedirs(anchor_save_path)

    def cluster(cluster_obj: KMeans, data: np.ndarray):
        """
        Cluster using joblib to process a batch in parallel.
        """
        clustering_op = cluster_obj.fit(data)
        return clustering_op

    def filter_trajs_end_pts(train_loader, train_trajs, train_end_pts,
                             train_attrs):
        for batch in tqdm(train_loader, desc="Processing data"):
            # [B, N]
            x_fut_padding_mask = batch['x_padding_mask'][..., 50:].any(-1)
            y = batch['y']
            x_attr_valid = batch['x_attr'][~x_fut_padding_mask]
            y_valid = y[~x_fut_padding_mask]  # [num_valid_agents, T, 2]
            y_end_pts_valid = y_valid[..., -1, :]  # [num_valid_agents, 2]
            train_trajs.append(y_valid)
            train_end_pts.append(y_end_pts_valid)
            train_attrs.append(x_attr_valid)

    def cluster_trajs_end_pts(trajs,
                              end_pts,
                              pre_fix,
                              k=32,
                              viz=True,
                              para=False):
        print(f"[INFO]: Clustering {pre_fix} trajs and end_pts.")
        if not para:
            end_pts_clustering = KMeans(n_clusters=k, verbose=1).fit(end_pts)
        else:
            n_cores = cpu_count()
            n_data = len(end_pts)
            batch_size = n_data // n_cores
            data_batches = [
                end_pts[i * batch_size:(i + 1) * batch_size]
                for i in range(n_cores)
            ]
            data_batches[-1] = end_pts[(n_cores - 1) * batch_size:]
            cluster_objs = [
                KMeans(n_clusters=k, verbose=1) for _ in range(n_cores)
            ]
            end_pts_clustering = Parallel(n_jobs=-1)(
                delayed(cluster)(cluster_objs[i], data_batches[i])
                for i in range(n_cores))
            end_pts_clustering = np.concatenate(end_pts_clustering)
        end_pts_anchors = np.zeros((k, 2))
        for i in range(k):
            end_pts_anchors[i] = np.mean(
                end_pts[end_pts_clustering.labels_ == i], axis=0)
        save_file = os.path.join(anchor_save_path,
                                 pre_fix + f"_anchor_end_pts_{k}.npy")
        print(f"[INFO]: Save path: {save_file}")
        np.save(save_file, end_pts_anchors)

        ds_size = trajs.shape[0]

        if not para:
            clustering = KMeans(n_clusters=k,
                                verbose=1).fit(trajs.reshape((ds_size, -1)))
        else:
            n_cores = cpu_count()
            n_data = ds_size
            batch_size = n_data // n_cores
            data_batches = [
                trajs[i * batch_size:(i + 1) * batch_size]
                for i in range(n_cores)
            ]
            data_batches[-1] = trajs[(n_cores - 1) * batch_size:]
            cluster_objs = [
                KMeans(n_clusters=k, verbose=1) for _ in range(n_cores)
            ]
            clustering = Parallel(n_jobs=-1)(
                delayed(cluster)(cluster_objs[i], data_batches[i])
                for i in range(n_cores))
            clustering = np.concatenate(clustering)
        anchors = np.zeros((k, 60, 2))
        for i in range(k):
            anchors[i] = np.mean(trajs[clustering.labels_ == i], axis=0)
        save_file = os.path.join(anchor_save_path,
                                 pre_fix + f"_anchor_trajs_{k}.npy")
        print(f"[INFO]: Save path: {save_file}")
        np.save(save_file, anchors)

        if viz:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            # Visualize all trajectories using green color and alpha=0.5
            for traj in trajs[:1000]:
                axs[0].plot(traj[:, 0], traj[:, 1], color='grey', alpha=0.5)
                axs[1].plot(traj[:, 0], traj[:, 1], color='grey', alpha=0.5)

            # Visualize all trajectory anchors on all trajectories
            for anchor in anchors:
                axs[0].plot(anchor[:, 0], anchor[:, 1], 'r*')

            axs[0].set_title('Trajectories and Anchors')

            # Visualize all endpoint anchors on all trajectories
            for end_pt in end_pts_anchors:
                axs[1].plot(end_pt[0], end_pt[1], 'b*')

            axs[1].set_title('Endpoint Anchors')

            # Save the figure to a file
            save_file = os.path.join(anchor_save_path,
                                     pre_fix + f"_clusters_{k}.png")
            plt.savefig(save_file)

    train_val_data = Av2DataModule(
        data_root=data_root,
        data_folder="multiagent-baseline-norm",
        num_workers=128,
        train_batch_size=128,
        val_batch_size=128,
        test_batch_size=128,
    )
    train_val_data.setup()

    print("load_data module...")
    train_trajs, train_end_pts, train_attrs = [], [], []
    val_trajs, val_end_pts, val_attrs = [], [], []
    test_trajs, test_end_pts, test_attrs = [], [], []

    # Create data loaders
    train_loader = train_val_data.train_dataloader()
    val_loader = train_val_data.val_dataloader()

    # ============== Val ================
    filter_trajs_end_pts(val_loader, val_trajs, val_end_pts, val_attrs)
    val_trajs = torch.concat(val_trajs, dim=0).numpy()
    val_end_pts = torch.concat(val_end_pts, dim=0).numpy()
    val_attrs = torch.concat(val_attrs, dim=0).numpy()
    pre_fix = "val"
    cluster_trajs_end_pts(val_trajs, val_end_pts, pre_fix=pre_fix)
    for i in range(3):
        cls_mask = (val_attrs[:, -1] == i)
        cluster_trajs_end_pts(val_trajs[cls_mask],
                              val_end_pts[cls_mask],
                              pre_fix=pre_fix + f"_{TYPE_MAP[i]}")

# ============== Train ================
    filter_trajs_end_pts(train_loader, train_trajs, train_end_pts, train_attrs)
    train_trajs = torch.concat(train_trajs, dim=0).numpy()
    train_end_pts = torch.concat(train_end_pts, dim=0).numpy()
    train_attrs = torch.concat(train_attrs, dim=0).numpy()
    pre_fix = "train"
    cluster_trajs_end_pts(train_trajs, train_end_pts, pre_fix=pre_fix)
    for i in range(3):
        cls_mask = (train_attrs[:, -1] == i)
        cluster_trajs_end_pts(train_trajs[cls_mask],
                              train_end_pts[cls_mask],
                              pre_fix=pre_fix + f"_{TYPE_MAP[i]}")

    # ============== Test ================
    train_val_data.test = True
    train_val_data.setup()
    test_loader = train_val_data.test_dataloader()
    filter_trajs_end_pts(test_loader, test_trajs, test_end_pts, test_attrs)
    test_trajs = torch.concat(test_trajs, dim=0).numpy()
    test_end_pts = torch.concat(test_end_pts, dim=0).numpy()
    test_attrs = torch.concat(test_attrs, dim=0).numpy()
    pre_fix = "test"
    cluster_trajs_end_pts(test_trajs, test_end_pts, pre_fix=pre_fix)
    for i in range(3):
        cls_mask = (test_attrs[:, -1] == i)
        cluster_trajs_end_pts(test_trajs[cls_mask],
                              test_end_pts[cls_mask],
                              pre_fix=pre_fix + f"_{TYPE_MAP[i]}")

    # ============== All ================
    all_trajs = np.concatenate([train_trajs, val_trajs, test_trajs], axis=0)
    all_end_pts = np.concatenate([train_end_pts, val_end_pts, test_end_pts],
                                 axis=0)
    all_attrs = np.concatenate([train_attrs, val_attrs, test_attrs], axis=0)
    pre_fix = "all"
    cluster_trajs_end_pts(all_trajs, all_end_pts, pre_fix=pre_fix)
    for i in range(3):
        cls_mask = (all_attrs[:, -1] == i)
        cluster_trajs_end_pts(all_trajs[cls_mask],
                              all_end_pts[cls_mask],
                              pre_fix=pre_fix + f"_{TYPE_MAP[i]}")

    print("load data done... \n All K-Means cluster done.")
