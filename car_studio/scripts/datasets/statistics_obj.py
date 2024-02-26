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
from nerfstudio.utils.io import load_from_json, write_to_json
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

@dataclass
class StastisticsObj:
    """count the obj numbers and sort."""
    data_dir: Path = Path('./data/car_studio/')
    """Path to dataset."""

    def main(self) -> None:
        name_dict = {}
        yaw_list = []
        for dataset_name in ['vk', 'ns', 'km', 'ko']:
            meta_all = load_from_json(self.data_dir / f'{dataset_name}.json')['instances']
            for meta in meta_all:
                obj_name = self._get_obj_id(meta['patch'][len('patch/'):])
                if obj_name in name_dict.keys():
                    name_dict[obj_name].append(meta['yaw'])
                else:
                    name_dict[obj_name] = [meta['yaw']]
        for key, values in name_dict.items():
            yaw_list.append({'key': key, 'yaws': values, 'yaw': self._get_yaw_std(values)})
        
        yaw_list.sort(key=lambda x: x['yaw'], reverse=True)
        idx = 0
        for i in yaw_list:
            if 'km' in i['key']:
                print(i)
                idx += 1
                if idx > 20:
                    break

        print(idx)

        # for file in tqdm(self.data_dir.iterdir()):
        #     obj_name = self._get_obj_id(file.name)
        #     if obj_name in name_dict.keys():
        #         name_dict[obj_name].append(file.name)
        #     else:
        #         name_dict[obj_name] = [file.name]
        # name_dict = dict(sorted(name_dict.items(), key=lambda item: len(item[1]), reverse=True))
        # # write_to_json('./name_dict.json', name_dict)
        # # out_dir = Path('./patches')
        # cnt = 0
        # for key, value in name_dict.items():
        #     # if 'ns' not in key:
        #     #     continue
        #     to_out_dir = Path(out_dir / key)
        #     if to_out_dir.exists():
        #         continue
        #     cnt += 1
        #     if cnt > 20:
        #         break
        #     for val in value:
        #         to_out_dir.mkdir(parents=True, exist_ok=True)
        #         Path(to_out_dir / val).symlink_to(Path(self.data_dir / val).absolute())


    def _get_obj_id(self, filename: str)-> str:
        """get object if from the file name"""
        dataset_name = filename[:2]
        name_parser_map = {
            'km': {
                'sid':[3,5],
                'oid':[10,13],
            },
            'vk': {
                'sid':[3,5],
                'oid':[10,13],
            },
            'ko': {
                'sid':[3,5],
                'oid':[10,15],
            },
            'ns': {
                'sid':[3,6],
                'oid':[11,16],
            }
        }
        sid_range = name_parser_map[dataset_name]['sid']
        sid = filename[sid_range[0]:sid_range[1]]
        oid_range = name_parser_map[dataset_name]['oid']
        oid = filename[oid_range[0]:oid_range[1]]
        return f'{dataset_name}{sid}{oid}'
    
    def _get_yaw_std(self, yaws):
        """https://stackoverflow.com/questions/13928404/calculating-standard-deviation-of-angles"""
        if len(yaws) == 1:
            return 0
        angles = np.array(yaws)
        sin_angles = np.sin(angles)
        cos_angles = np.cos(angles)

        mean_sin = np.mean(sin_angles)
        mean_cos = np.mean(cos_angles)

        std = np.sqrt(-np.log(mean_sin ** 2 + mean_cos ** 2))
        return std


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(StastisticsObj).main()


if __name__ == "__main__":
    entrypoint()
