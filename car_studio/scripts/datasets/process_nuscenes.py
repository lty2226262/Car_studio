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
from typing import Literal, Tuple

import imageio
import numpy as np
import torch
import tyro
from nerfstudio.utils.io import write_to_json
from nuscenes.nuscenes import NuScenes as NuScenesDatabase
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from tqdm import tqdm

from car_studio.utils.kitti_utils import collate_fn_dataloader
from car_studio.utils.nuscenes_utils import ProcessNuscenes


@dataclass
class ProcessNuScenesMasks:
    """Use cuboid detections to render masks for dynamic objects."""

    data_dir: Path = Path("data/nuscenes")
    """Path to NuScenes dataset."""
    version: Literal["v1.0-mini", "v1.0-trainval"] = "v1.0-trainval"
    """Which version of the dataset to process."""
    cameras: Tuple[Literal["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT",
                           "BACK_RIGHT"], ...] = (
        "FRONT",
        "FRONT_LEFT",
        "FRONT_RIGHT",
        "BACK",
        "BACK_LEFT",
        "BACK_RIGHT",
    )
    """Which cameras to use."""
    car_type: Tuple[Literal['vehicle.car', 'vehicle.construction', 'vehicle.truck'], ...] = ('vehicle.car')
    """Car type to process, 'vehicle.car', 'vehicle.construction', 'vehicle.truck' are valid"""
    dataset_type: Literal['ns'] = 'ns'
    """Dataset type ns: nuscenes"""
    output_dir: Path = Path('./data/car_studio/')
    """Path to the output directory."""

    def main(self) -> None:
        """Generate NuScenes dynamic object masks."""

        nusc = NuScenesDatabase(
            version=self.version,
            dataroot=str(self.data_dir.absolute()),
        )
        cameras = ["CAM_" + camera for camera in self.cameras]

        # get samples for scene
        samples = [samp for samp in nusc.sample]

        # sort by timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))
        coarse_instances_list = []

        sid = fid = oid = 0
        seq_map_dict = dict()
        frame_map_dict = dict()
        obj_map_dict = dict()

        for sample in tqdm(samples):
            seq_token = sample['scene_token']
            frame_token = sample['token']
            if seq_token not in seq_map_dict.keys():
                seq_map_dict[seq_token] = sid
                sid += 1
            if frame_token not in frame_map_dict.keys():
                frame_map_dict[frame_token] = fid
                fid +=1
            for camera in cameras:
                camera_data = nusc.get("sample_data", sample["data"][camera])

                _, boxes, intrinsics = nusc.get_sample_data(sample["data"][camera], box_vis_level=BoxVisibility.ANY)
                boxes = [i for i in boxes if i.name in self.car_type]
                if len(boxes) > 0:
                    pass
                else:
                    continue

                file_path = camera_data['filename']
                calibrated_sensor = nusc.get("calibrated_sensor", camera_data["calibrated_sensor_token"])

                for box in boxes:
                    instance_token = nusc.get("sample_annotation", box.token)["instance_token"]
                    if instance_token not in obj_map_dict.keys():
                        obj_map_dict[instance_token] = oid
                        oid += 1
                    corners_3d = box.corners()
                    corners = view_points(corners_3d, intrinsics, normalize=True)[:2, :]
                    corners = np.round(corners).astype(int).T

                    height = int(camera_data['height'])
                    width = int(camera_data['width'])

                    xmin, ymin = np.min(corners, axis=0)

                    xmin = int(np.clip(xmin,0,width - 1))
                    ymin = int(np.clip(ymin,0,height - 1))
                    
                    xmax, ymax = np.max(corners, axis=0)

                    xmax = int(np.clip(xmax,0,width - 1))
                    ymax = int(np.clip(ymax,0,height - 1))

                    pre_rot_mat = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
                    rot_mat = np.dot(pre_rot_mat, box.rotation_matrix)
                    yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])

                    file_name = f'{self.dataset_type}_{seq_map_dict[seq_token]:03d}{frame_map_dict[frame_token]:05d}{obj_map_dict[instance_token]:05d}'
                    patch_file_name = file_name + '_patch.png'
                    mask_file_name = file_name + '_mask.png'

                    one_patch = {'patch' : 'patch/' + patch_file_name,
                                'mask' : 'mask/' + mask_file_name,
                                'image_file' : '../nuscenes/' + file_path,
                                'fl_x' : calibrated_sensor['camera_intrinsic'][0][0],
                                'fl_y' : calibrated_sensor['camera_intrinsic'][0][1],
                                'cx' : calibrated_sensor['camera_intrinsic'][0][2],
                                'cy' : calibrated_sensor['camera_intrinsic'][1][2],
                                'cam_tx' : 0.0,
                                'cam_ty' : 0.0,
                                'cam_tz' : 0.0,
                                'xmin': xmin,
                                'xmax': xmax,
                                'ymin': ymin,
                                'ymax': ymax,
                                'height': box.wlh[2],
                                'width': box.wlh[0],
                                'length': box.wlh[1],
                                'obj_x': box.center[0],
                                'obj_y': box.center[1],
                                'obj_z': box.center[2],
                                'yaw': yaw,
                                'ns' : {
                                    'scene_token': seq_token,
                                    'sample_token': frame_token,
                                    'instance_token': instance_token,
                                    }
                                }
                    coarse_instances_list.append(one_patch)

        dataset_to_process = ProcessNuscenes(dataset_dir=self.data_dir,
                                                 dataset_type=self.dataset_type,
                                                 dict_list=coarse_instances_list)
        loader = torch.utils.data.DataLoader(
            dataset_to_process,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn = collate_fn_dataloader
        )

        json_file = {}
        json_file['dataset_name'] = self.dataset_type
        instances_list = []

        for batch in tqdm(loader):
            patch, masked_patch, json_seg = batch
            if len(json_seg) == 0:
                continue
            json_item = json_seg[0]
            patch_path = self.output_dir.joinpath(json_item['patch']).absolute()
            mask_path = self.output_dir.joinpath(json_item['mask']).absolute()
            os.makedirs(str(mask_path.parent), exist_ok=True)
            os.makedirs(str(patch_path.parent), exist_ok=True)
            imageio.v2.imwrite(str(patch_path), patch[0])
            imageio.v2.imwrite(str(mask_path),
                                np.repeat(np.array(masked_patch[0] * 255,
                                                dtype=np.uint8), 3, axis=-1))
            instances_list.append(json_item)

        json_file['instances'] = instances_list
        instance_file_path = self.output_dir.joinpath(f'{self.dataset_type}.json')
        write_to_json(instance_file_path, json_file)

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ProcessNuScenesMasks).main()


if __name__ == "__main__":
    entrypoint()
