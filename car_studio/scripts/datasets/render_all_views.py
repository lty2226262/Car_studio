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

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type

import imageio
import numpy as np
import torch
import torchvision.transforms as T
import tyro
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import (CAMERA_MODEL_TO_TYPE, Cameras,
                                        CameraType)
from nerfstudio.data.dataparsers.base_dataparser import (DataParser,
                                                         DataParserConfig,
                                                         DataparserOutputs)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.utils.io import load_from_json, write_to_json
from nerfstudio.utils.math import intersect_aabb
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from car_studio.data.utils.data_utils import get_image_tensor_from_path
from car_studio.models.car_nerf import CarNerfModel, CarNerfModelConfig
from car_studio.utils.kitti_utils import (ProcessKittiMot, ProcessKittiObj,
                                          ProcessVKitti, collate_fn_dataloader)


@dataclass
class RenderAllViews:
    """render all views."""
    data: Path = Path('./data/car_studio/')
    """Path to dataset."""
    subset: Tuple[Literal["vk", "ko", "ns", "km"], ...] = ("km")
    """Dataset to load."""
    obj_list: Tuple[str] = ("km07027", "km09030", "km09029")
    """Object id lists."""
    store_original: bool = True
    """Copy the original file to the destination dir and append the info to the generated json."""
    output: Path = Path('./render_out/')
    """Output path directory."""
    model_checkpoint: Path = Path('./render_out/step-000787000.ckpt')
    """model checkpoint"""

    def main(self) -> None:
        """render all views"""
        assert self.data.exists(), f"Data directory {self.data} does not exist."
        if isinstance(self.subset, str):
            self.subset = tuple([self.subset])
        if isinstance(self.obj_list, str):
            self.obj_list = tuple([self.obj_list])
        assert self.model_checkpoint.exists(), f"Checkpoint {self.model_checkpoint} does not exist"
        preload_latents = {}
        scene_box = SceneBox(torch.tensor([[-1., -1., -1.],
                            [1.0, 1.0, 1.0]]))
        device = 'cuda'
        self.model = CarNerfModel(scene_box=scene_box,
                                  num_train_data=114514,
                                  config=CarNerfModelConfig(background_color='black')).to(device)
        self.img_transform = T.Compose([T.Resize((128, 128)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        loaded_state = torch.load(self.model_checkpoint, map_location='cpu')['pipeline']
        model_state = {}
        for key, value in loaded_state.items():
            if key.startswith('_model.'):
                model_state[key[len("_model."):]] = value
            else:
                CONSOLE.print(f"[bold yellow]Warning: unrecognized key {key}")
        self.model.load_state_dict(model_state,strict=True)

        meta_list = []
        all_instances_dict = {}
        for sub_set in self.subset:
            meta = load_from_json(self.data / (sub_set + ".json"))
            meta_list.append(meta)
            for instance_it in meta['instances']:
                obj_id = self._get_obj_id(instance_it['patch'][6:])
                if obj_id in self.obj_list:
                    if obj_id in all_instances_dict.keys():
                        all_instances_dict[obj_id].append(instance_it)
                    else:
                        all_instances_dict[obj_id] = [instance_it]
        skip_obj = []
        render_instances = []

        for obj_it in self.obj_list:
            all_instances_dict[obj_it] = sorted(all_instances_dict[obj_it],
                                                key=lambda x:abs(10.0 - x['obj_z']))
            nearest_element = all_instances_dict[obj_it][0]
            tx = 0.0
            ty = nearest_element['obj_y']
            tz = 10.0
            l = nearest_element['length']
            h = nearest_element['height']
            w = nearest_element['width']
            imshape_h = nearest_element['h']
            imshape_w = nearest_element['w']
            yaws = np.linspace(-np.pi, np.pi, num=20, endpoint=False)
            P = np.array([[nearest_element['fl_x'], 0, nearest_element['cx'], nearest_element['cam_tx']],
                        [0, nearest_element['fl_y'], nearest_element['cy'], nearest_element['cam_ty']],
                        [0, 0, 1, nearest_element['cam_tz']]])
            if obj_it not in preload_latents.keys():
                latents = self._get_latents(self.data / nearest_element['patch'], device)
                preload_latents[obj_it] = latents
            else:
                latents = preload_latents[obj_it].to(device)
            for idx, ry in tqdm(enumerate(yaws)):
                corners = self._get_corners([tx, ty, tz, l, h, w, ry])
                pixels = self._project_to_image(corners, P)
                xmin, ymin = np.min(pixels.astype(int), axis=0)
                xmax, ymax = np.max(pixels.astype(int), axis=0)
                assert 0 <= xmin <= imshape_w and 0 <= xmax <= imshape_w
                assert 0 <= ymin <= imshape_h and 0 <= ymax <= imshape_h
                new_instance = nearest_element.copy()
                new_instance['patch'] = f'patch/{obj_it}_{idx:02d}_patch.png'
                new_instance['mask'] = f'mask/{obj_it}_{idx:02d}_mask.png'
                new_instance['image_file'] = 'empty'
                new_instance['xmin'] = int(xmin)
                new_instance['xmax'] = int(xmax)
                new_instance['ymin'] = int(ymin)
                new_instance['ymax'] = int(ymax)
                new_instance['obj_x'] = tx
                new_instance['obj_y'] = ty
                new_instance['obj_z'] = tz
                new_instance['yaw'] = ry
                render_instances.append(new_instance)
                out_img = self._get_output_images(new_instance=new_instance, 
                                                  latents=latents,
                                                  device=device)
                self._save_img(out_img['rgb_fine'],
                               self.output / f'{obj_it}/{new_instance["patch"]}')
                self._save_img(out_img['accumulation_fine'].squeeze(-1),
                               self.output / f'{obj_it}/{new_instance["mask"]}')
            # count unprocessed elements
            if obj_it not in all_instances_dict.keys():
                skip_obj.append(obj_it)
            
            if self.store_original:
                for instance_it in all_instances_dict[obj_it]:
                    for iter_name in ['patch', 'mask']:
                        src = Path(self.data / instance_it[iter_name]).absolute()
                        dst = Path(self.output / f'{obj_it}/{instance_it[iter_name]}')
                        if not dst.exists():
                            dst.symlink_to(src)
                    render_instances.append(instance_it)
                
                write_to_json(Path(self.output / f'{obj_it}/transforms.json'), {'instance':render_instances})
        CONSOLE.print(f"[bold yellow]Warning: Skip {len(skip_obj)} objects: {skip_obj}")
                
        
        # name_dict = {}
        # for file in tqdm(self.data_dir.iterdir()):
        #     obj_name = self._get_obj_id(file.name)
        #     if obj_name in name_dict.keys():
        #         name_dict[obj_name].append(file.name)
        #     else:
        #         name_dict[obj_name] = [file.name]
        # name_dict = dict(sorted(name_dict.items(), key=lambda item: len(item[1]), reverse=True))
        # # write_to_json('./name_dict.json', name_dict)
        # out_dir = Path('./patches')
        # cnt = 0
        # for key, value in name_dict.items():
        #     if 'ns' not in key:
        #         continue
        #     to_out_dir = Path(out_dir / key)
        #     if to_out_dir.exists():
        #         continue
        #     cnt += 1
        #     if cnt > 20:
        #         break
        #     for val in value:
        #         to_out_dir.mkdir(parents=True, exist_ok=True)
        #         Path(to_out_dir / val).symlink_to(Path(self.data_dir / val).absolute())
        return 0


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

    def _get_corners(self, obj):
        if isinstance(obj, list):
            tx, ty, tz, l, h, w, ry = obj
        else:
            tx, ty, tz, l, h, w, ry = list(obj)
        
        # 3d bounding box corners
        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
        y_corners = [0,0,0,0,-h,-h,-h,-h]
        z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    
        R = self._roty(ry)    
        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
        #print corners_3d.shape
        corners_3d[0,:] = corners_3d[0,:] + tx
        corners_3d[1,:] = corners_3d[1,:] + ty
        corners_3d[2,:] = corners_3d[2,:] + tz
        return np.transpose(corners_3d)
    
    def _roty(self, t):
        ''' Rotation about the y-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                        [0,  1,  0],
                        [-s, 0,  c]])
    
    def _project_to_image(self, pts_3d, P):
        ''' Project 3d points to image plane.

        Usage: pts_2d = projectToImage(pts_3d, P)
        input: pts_3d: nx3 matrix
                P:      3x4 projection matrix
        output: pts_2d: nx2 matrix

        P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
        => normalize projected_pts_2d(2xn)

        <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
            => normalize projected_pts_2d(nx2)
        '''
        n = pts_3d.shape[0]
        pts_3d_extend = np.hstack((pts_3d, np.ones((n,1))))
        # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
        pts_2d = np.dot(pts_3d_extend, np.transpose(P)) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]

    def _get_latents(self, patch_image:Path, device)->torch.Tensor:
        with torch.no_grad():
            image_tensor = get_image_tensor_from_path(patch_image)
            image_tensor = self.img_transform(image_tensor.permute(2,0,1)).permute(1,2,0).to(device)
            return self.model.get_latents(image_tensor.permute(2,0,1).unsqueeze(0))
        
    def _get_pose(self, instance_it: Dict) -> List:
        height = instance_it['height']
        width = instance_it['width']
        length = instance_it['length']
        obj_x = instance_it['obj_x']
        obj_y = instance_it['obj_y']
        obj_z = instance_it['obj_z']
        yaw = instance_it['yaw']

        pose_w = np.array([obj_x, obj_y, obj_z]).copy()
        dim = np.array([length, height, width]).copy()
        pose_w[1] -= dim[1] / 2.0

        trans_mat = -pose_w

        rot_mat = self._roty(yaw).T

        pose_inv = np.dot(rot_mat, trans_mat)

        # pose = np.linalg.inv(pose_inv)
        pose = np.identity(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = pose_inv
        
        scaler = np.diag(np.append(1.0 / (dim / 2.0 + 1e-10), 1.0))
        scaling_pose = np.dot(scaler, pose)

        cam_vir2cam = np.diag([1., -1., -1., 1.])
        scaling_pose = np.dot(scaling_pose, cam_vir2cam)

        return scaling_pose

    @torch.no_grad()
    def _get_output_images(self, latents:torch.Tensor, new_instance:Dict, device):
        pose = self._get_pose(new_instance)
        new_camera = Cameras(
            fx=float(new_instance['fl_x']),
            fy=float(new_instance['fl_y']),
            cx=float(new_instance['cx']),
            cy=float(new_instance['cy']),
            distortion_params=camera_utils.get_distortion_params(),
            height=new_instance['h'],
            width=new_instance['w'],
            camera_type=CameraType.PERSPECTIVE,
            camera_to_worlds=torch.from_numpy(pose[None, :3, :4]).to(torch.float),
        ).to(device)
        img_whole_coords = new_camera.get_image_coords()
        xyxy_msk = torch.from_numpy(np.array([new_instance["xmin"],
                                new_instance["ymin"],
                                new_instance["xmax"],
                                new_instance["ymax"]])).to(torch.long)
        xmin, ymin, xmax, ymax = xyxy_msk
        img_coords = img_whole_coords[ymin:ymax+1, xmin:xmax+1,:]
        ray_bundle = new_camera.generate_rays(0 ,coords=img_coords).to(torch.float)
        scene_box = torch.Tensor([-1., -1., -1., 1., 1., 1.]).to(device)
        t_min, t_max = intersect_aabb(ray_bundle.origins,
                                ray_bundle.directions,
                                scene_box)
        t_min = t_min.unsqueeze(-1)
        t_max = t_max.unsqueeze(-1)

        ray_bundle.nears = t_min
        ray_bundle.fars = t_max
        ray_bundle.metadata['latents'] = latents
        ray_bundle.metadata['xyxy_mask'] = xyxy_msk
        img = self.model.get_outputs_for_camera_ray_bundle_and_latents(ray_bundle)
        return img

    def _save_img(self, img: torch.Tensor, save_path: Path) -> None:
        if len(img.shape) == 2:
           pass 
        elif len(img.shape) == 3:
            assert(img.shape)[-1] == 3
            img = img.permute(2,0,1)
        else:
            assert False, 'invalid input image format'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(img, str(save_path))



def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderAllViews).main()


if __name__ == "__main__":
    entrypoint()
