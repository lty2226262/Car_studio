# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import imageio
import numpy as np
import torch
import tyro
from nerfstudio.utils.io import write_to_json
from PIL import Image
from tqdm import tqdm

from car_studio.utils.kitti_utils import (ProcessKittiMot, ProcessKittiObj,
                                          ProcessVKitti, collate_fn_dataloader)


@dataclass
class ProcessNuScenesMasks:
    """Use cuboid detections to render masks for dynamic objects."""

    data_dir: Path = Path('./data/kitti-obj/')
    """Path to dataset."""
    dataset: Literal["vk", "ko", "km"] = "ko"
    """Which dataset to process. vk: vkitti, ko: kitti-obj, km: kitti-mot"""
    output_dir: Path = Path('./data/car_studio/')
    """Path to the output directory."""

    def main(self) -> None:
        """Generate patches, masks and intrinsics for kitti datasets."""
        if self.dataset == 'vk':
            dataset_to_process = ProcessVKitti(dataset_dir=self.data_dir,
                                dataset_type=self.dataset)
        elif self.dataset == 'km':
            dataset_to_process = ProcessKittiMot(dataset_dir=self.data_dir,
                                                 dataset_type=self.dataset)
        elif self.dataset == 'ko':
            dataset_to_process = ProcessKittiObj(dataset_dir=self.data_dir,
                                                 dataset_type=self.dataset)
        loader = torch.utils.data.DataLoader(
            dataset_to_process,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn = collate_fn_dataloader
            )
        
        json_file = dict()
        json_file['dataset_name'] = self.dataset
        instances_list = list()

        for batch in tqdm(loader):
            patch, masked_patch, json_seg = batch
            for idx, json_item in enumerate(json_seg):
                patch_path = self.output_dir.joinpath(json_item['patch']).absolute()
                mask_path = self.output_dir.joinpath(json_item['mask']).absolute()
                os.makedirs(str(mask_path.parent), exist_ok=True)
                os.makedirs(str(patch_path.parent), exist_ok=True)
                imageio.v2.imwrite(str(patch_path), patch[idx])
                Image.fromarray(np.array(masked_patch[idx] * 255, dtype=np.uint8).squeeze(-1)).save(str(mask_path))
                instances_list.append(json_item)
        json_file['instances'] = instances_list
        instance_file_path = self.output_dir.joinpath(f'{self.dataset}.json')
        write_to_json(instance_file_path, json_file)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ProcessNuScenesMasks).main()


if __name__ == "__main__":
    entrypoint()
