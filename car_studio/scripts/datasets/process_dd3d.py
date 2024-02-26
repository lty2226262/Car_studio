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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
import tyro
from nerfstudio.utils.io import load_from_json, write_to_json
from PIL import Image
from pyquaternion import Quaternion
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from car_studio.utils.kitti_utils import (ProcessKittiMot, ProcessKittiObj,
                                          ProcessVKitti, collate_fn_dataloader)


class DD3DDatasets(Dataset):
    def __init__(self, root_dir : Path, json_file: list):
        self.root_dir = root_dir
        self.json_file = json_file

    def __len__(self):
        return len(self.json_file)
    
    def __getitem__(self, index):
        datum = {}
        instance = self.json_file[index]
        file_path = str((self.root_dir / instance['image_file']).absolute())
        img = Image.open(file_path)
        datum['image'] = np.array(img)
        datum['xyxy_bbox'] = np.array([instance['xmin'], instance['ymin'], instance['xmax'], instance['ymax']], dtype=int)
        datum['idx'] = index
        return datum

@dataclass
class ProcessDD3D:
    """count the obj numbers and sort."""
    dvm_data_dir: Path = Path('./data/dvm_cars/')
    """Path to dataset."""
    filtered_json: Path = Path('./outputs/filter_dd3d/20230615/115304/filtered_result.json')
    output_dir: Path = Path('./data/car_studio/')

    def main(self) -> None:
        this_json = load_from_json(self.filtered_json)
        assert this_json['dataset_name'] == 'dvm_car_studio'
        datasets = DD3DDatasets(self.dvm_data_dir, this_json['instances'])
        dataloader = DataLoader(datasets, batch_size=None,
                                pin_memory=True, collate_fn=None)

        sam_checkpoint = "dependencies/segment-anything/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        json_segment = list()
        obj_id_dict = {}
        cnt = 0
        obj_id_num = 0
        frame_id = 0


        for sample in tqdm(dataloader):
            image = sample['image'].cpu().numpy()
            predictor.set_image(image)
            xmin, ymin, xmax, ymax = sample['xyxy_bbox'].to(device)
            transformed_boxes = predictor.transform.apply_boxes_torch(sample['xyxy_bbox'].to(device), tuple(image.shape[:2]))
            masks, ious, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            intersec = masks[0].sum() / ((xmax + 1 - xmin) * (ymax + 1 - ymin))
            if float(ious[0]) > 0.6 and intersec.item() > 0.6:
                mask_to_save = masks[0, :, ymin: ymax + 1,
                                    xmin: xmax +1].permute((1, 2, 0)).cpu().numpy()
                patch_to_save = image[ymin: ymax + 1, xmin: xmax + 1, :]
                info = this_json['instances'][sample['idx']]
                obj_id = info['image_file'].rsplit('$', maxsplit=1)[0].rsplit('/', maxsplit=1)[-1]
                if obj_id in obj_id_dict.keys():
                    this_obj_num = obj_id_dict[obj_id]
                else:
                    obj_id_dict[obj_id] = obj_id_num
                    this_obj_num = obj_id_num
                    obj_id_num += 1
                filename = f'dv_0{frame_id:06d}{this_obj_num:06d}'
                frame_id += 1
                patch_file_name = filename + '_patch.png'
                mask_file_name = filename + '_mask.png'
                info['patch'] = 'patch/' + patch_file_name
                info['mask'] = 'mask/' + mask_file_name
                json_segment.append(info)
                patch_path = self.output_dir.joinpath(info['patch']).absolute()
                mask_path = self.output_dir.joinpath(info['mask']).absolute()
                os.makedirs(str(mask_path.parent), exist_ok=True)
                os.makedirs(str(patch_path.parent), exist_ok=True)
                imageio.v2.imwrite(str(patch_path), patch_to_save)
                Image.fromarray(np.array(mask_to_save * 255, dtype=np.uint8).squeeze(-1)).save(str(mask_path))
            else:
                cnt += 1
        print(f'skip {cnt} pictures')
        json_file = dict()
        json_file['dataset_name'] = 'dvm_car_studio'
        json_file['instances'] = json_segment
        instance_file_path = self.output_dir.joinpath('dv.json')
        write_to_json(instance_file_path, json_file)

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ProcessDD3D).main()


if __name__ == "__main__":
    entrypoint()
