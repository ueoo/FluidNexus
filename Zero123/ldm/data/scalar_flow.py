import copy
import csv
import json
import math
import os
import random
import sys

from pathlib import Path
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import webdataset as wds

from datasets import load_dataset
from einops import rearrange
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from ldm.util import instantiate_from_config


class ScalarFlowStaticDataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        batch_size,
        total_view,
        train=None,
        validation=None,
        test=None,
        num_workers=4,
        paths_post="",
        cond_view=-1,
        target_view=-1,
        white_bg=False,
        **kwargs,
    ):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view
        self.paths_post = paths_post
        self.cond_view = cond_view
        self.target_view = target_view
        self.white_bg = white_bg

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if "image_transforms" in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend(
            [transforms.ToTensor(), transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c"))]
        )
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        dataset = ScalarFlowStaticData(
            root_dir=self.root_dir,
            total_view=self.total_view,
            validation=False,
            image_transforms=self.image_transforms,
            paths_post=self.paths_post,
            cond_view=self.cond_view,
            target_view=self.target_view,
            white_bg=self.white_bg,
        )
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler
        )

    def val_dataloader(self):
        dataset = ScalarFlowStaticData(
            root_dir=self.root_dir,
            total_view=self.total_view,
            validation=True,
            image_transforms=self.image_transforms,
            paths_post=self.paths_post,
            cond_view=self.cond_view,
            target_view=self.target_view,
            white_bg=self.white_bg,
        )
        # sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        dataset = ScalarFlowStaticData(
            root_dir=self.root_dir,
            total_view=self.total_view,
            validation=self.validation,
            paths_post=self.paths_post,
            cond_view=self.cond_view,
            target_view=self.target_view,
            white_bg=self.white_bg,
        )
        return wds.WebLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class ScalarFlowStaticData(Dataset):
    def __init__(
        self,
        root_dir,
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=5,
        validation=False,
        paths_post="",
        cond_view=-1,
        target_view=-1,
        white_bg=False,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view
        self.cond_view = cond_view
        self.target_view = target_view
        self.white_bg = white_bg
        self.camera_root_dir = os.path.join(root_dir, "camera")

        # total_objects = len(self.paths)
        # # 104 simulation, 140 frames (20-160)
        # # use first 1.5 % simulation as validation
        # if validation:
        #     self.paths = self.paths[: math.floor(total_objects / 100.0 * 1.5)]  # used first 1% as validation
        # else:
        #     self.paths = self.paths[math.floor(total_objects / 100.0 * 1.5) :]  # used last 99% as training

        if not validation:
            self.paths_json = os.path.join(root_dir, f"train_paths{paths_post}.json")
        else:
            self.paths_json = os.path.join(root_dir, f"val_paths{paths_post}.json")
        assert os.path.exists(self.paths_json), f"{self.paths_json} does not exist."
        assert os.path.exists(self.camera_root_dir), f"{self.camera_root_dir} does not exist."

        print(f"============= loading dataset =============")
        print(f"root_dir: {root_dir}")
        print(f"total_view: {self.total_view}")
        print(f"cond_view: {self.cond_view}")
        print(f"target_view: {self.target_view}")
        print(f"white_bg: {self.white_bg}")
        print(f"paths_post: {paths_post}")
        print(f"paths_json: {self.paths_json}")
        print(f"===========================================")

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        with open(self.paths_json) as f:
            self.paths = json.load(f)

        print("============= length of dataset %d =============" % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def load_im(self, path):
        """
        replace background pixel with random color in rendering
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.white_bg:
            img = 255 - img
        img = Image.fromarray(img)
        return img

    def __getitem__(self, index):

        data = {}
        if 0 <= self.cond_view < self.total_view and 0 <= self.target_view < self.total_view:
            index_target = self.target_view
            index_cond = self.cond_view
        else:
            total_view = self.total_view
            index_target, index_cond = random.sample(range(total_view), 2)  # without replacement

        filename = os.path.join(self.root_dir, self.paths[index])

        # print(self.paths[index])

        if self.return_paths:
            data["path"] = str(filename)

        # color = [1.0, 1.0, 1.0, 1.0]

        target_im = self.process_im(self.load_im(os.path.join(filename, f"{index_target:02d}.png")))
        cond_im = self.process_im(self.load_im(os.path.join(filename, f"{index_cond:02d}.png")))
        target_RT = np.load(os.path.join(self.camera_root_dir, f"{index_target:02d}.npy"))
        cond_RT = np.load(os.path.join(self.camera_root_dir, f"{index_cond:02d}.npy"))

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_RT, cond_RT)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
