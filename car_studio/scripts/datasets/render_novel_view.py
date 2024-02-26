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

import copy
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
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.utils.io import load_from_json, write_to_json
from nerfstudio.utils.math import intersect_aabb
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from car_studio.data.utils.data_utils import get_image_tensor_from_path
from car_studio.models.car_nerf import CarNerfModel, CarNerfModelConfig
from car_studio.utils.kitti_utils import (Calibration, Object3d,
                                          ProcessKittiMot, ProcessKittiObj,
                                          ProcessVKitti, collate_fn_dataloader)


@dataclass
class RenderNovelView:
    """render all views."""
    data: Path = Path('./data/kitti-mot/')
    """Path to dataset."""
    output: Path = Path('./render_out_color_ko/')
    """Output path directory."""
    # model_checkpoint: Path = Path('./outputs/car-nerf-ko-sv-seq-1-optimization/car-nerf-stage-two/2023-06-26_151219/nerfstudio_models/step-000049999.ckpt')
    model_checkpoint: Path = Path('/home/joey/code/car-studio/outputs/stage-one-ko-sv-with-mask/car-nerf/2023-06-16_133708/nerfstudio_models/step-000500000.ckpt')
    """model checkpoint"""
    encoder_checkpoint: Path = Path('/home/joey/code/car-studio/outputs/stage-one-ko-sv-with-mask/car-nerf/2023-06-16_133708/nerfstudio_models/step-000500000.ckpt')
    """encoder checkpoint"""
    model_config: ModelConfig = CarNerfModelConfig(background_color='white')
    """model config"""
    encoder_config: ModelConfig = CarNerfModelConfig(background_color='white')
    """model config""" 
    sequence_num: int = 1
    """sequence id"""
    start_id: int = 73
    """Start id, -1 for the very start frame 73"""
    end_id: int = 138
    """End id, -1 for the very last frame 138"""
    rotated_frame: Tuple[int] = (97, )
    """rotated_frame"""
    trans_frame: Tuple[int] = (88, )
    """translation_frame"""
    trans_instance: Tuple[str] = ('km01020', )
    """trans_instance"""
    colors_frame: Tuple[int] = (113, )
    """color_frame"""
    color_instance: Tuple[str] = ('km01034', )
    """change color instance"""

    def main(self) -> None:
        """render all views"""
        device = 'cuda'
        self.img_transform = T.Compose([T.Resize((128, 128)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.model, self.global_latents_lookup_table = self._init_model_and_load_weigths(device)
        self.all_objects = self._parse_all_instances()
        self.views = self._build_trans()
        
        optimized_frames = {}
        unoptimized_frames = {}
        for obj_it in self.all_objects:
            fid = int(obj_it['patch'][11:16])
            obj_id = self._get_obj_id(obj_it)
            if obj_id in self.global_latents_lookup_table.keys():
                if fid in optimized_frames.keys():
                    optimized_frames[fid].append(obj_it)
                else:
                    optimized_frames[fid] = [obj_it]
            else:
                if fid in unoptimized_frames.keys():
                    unoptimized_frames[fid].append(obj_it)
                else:
                    unoptimized_frames[fid] = [obj_it]

        # self._generate_all_seq(optimized_frames,
        #                         unoptimized_frames,
        #                         device)
        
        # self._get_all_car_rotated(optimized_frames,
        #                           unoptimized_frames,
        #                           device)
        # self._get_one_car_translation(optimized_frames,
        #                               unoptimized_frames,
        #                               device)

        self._change_colors(optimized_frames,
                            unoptimized_frames,
                            device)
        
        return

    def _get_obj_id(self, instance_it: Dict)-> str:
        """get object if from the file name"""
        dataset_name = instance_it['patch'][6:8]
        assert dataset_name == 'km'
        name_parser_map = {
            'km': {
                'sid':[9,11],
                'oid':[16,19],
            },
        }
        sid_range = name_parser_map[dataset_name]['sid']
        sid = instance_it['patch'][sid_range[0]:sid_range[1]]
        oid_range = name_parser_map[dataset_name]['oid']
        oid = instance_it['patch'][oid_range[0]:oid_range[1]]
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
    def _get_output_images(self, latents:torch.Tensor, new_instance:Dict, device, trans = np.identity(4)):
        pose = np.dot(self._get_pose(new_instance), trans)
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

    def _init_model_and_load_weigths(self, device:str = 'cuda') -> Tuple[torch.nn.Module, Dict]:
        assert self.data.exists(), f"Data directory {self.data} does not exist."
        assert self.model_checkpoint.exists(), f"Checkpoint {self.model_checkpoint} does not exist"
        assert self.encoder_checkpoint.exists(), f"Checkpoint {self.model_checkpoint} does not exist"
        loaded_state = torch.load(self.model_checkpoint, map_location="cpu")
        loaded_state_encoder = torch.load(self.encoder_checkpoint, map_location="cpu")

        latents_lookup_table = {}
        state = {}
        for key, value in loaded_state['pipeline'].items():
            if key.startswith("module."):
                key = key[len("module:"):]
            if key.startswith('_model.'):
                key = key[len('_model.'):]
            if 'latent_vectors' in key:
                value = value.to(device)
                latents_lookup_table.update({key.rsplit('_', maxsplit=1)[-1]: value})
            else:
                state.update({key:value})
        for key, value in loaded_state_encoder['pipeline'].items():
            if key.startswith("module."):
                key = key[len("module:"):]
            if key.startswith('_model.'):
                key = key[len('_model.'):]
            if key.startswith('encoder'):
                state.update({key:value})

        scene_box = SceneBox(torch.tensor([[-1., -1., -1.],
                    [1.0, 1.0, 1.0]]))
        device = 'cuda'
        model = self.model_config.setup(
            scene_box=scene_box,
            num_train_data = 114514,
        ).to(device)

        model.load_state_dict(state, strict=True)

        return (model, latents_lookup_table)

    def _init_encoder(self, device:str = 'cuda') -> torch.nn.Module:
        assert self.data.exists(), f"Data directory {self.data} does not exist."
        assert self.encoder_checkpoint.exists(), f"Checkpoint {self.model_checkpoint} does not exist"
        new_loaded_state = torch.load(self.encoder_checkpoint, map_location="cpu")
        new_state = {}
        for key, value in new_loaded_state['pipeline'].items():
            if key.startswith("module."):
                key = key[len("module:"):]
            if key.startswith('_model.'):
                key = key[len('_model.'):]
            if not key.startswith('field'):
                new_state.update({key:value})
        scene_box = SceneBox(torch.tensor([[-1., -1., -1.],
                    [1.0, 1.0, 1.0]]))
        device = 'cuda'
        new_model = CarNerfModel(self.encoder_config, scene_box, num_train_data=114515).to(device)
        if hasattr(new_model, 'field'):
            del new_model.field

        new_model.load_state_dict(new_state, strict=True)

        return new_model
    
    def _parse_all_instances(self):
        label_path = str(self.data / f'training/label_02/{self.sequence_num:04d}.txt')
        with open(label_path, encoding='UTF-8') as f:
            lines = f.readlines()

        instance_list = []
        for line in lines:
            elements = line.split()
            fid = int(elements[0])
            if fid < self.start_id or fid > self.end_id:
                continue
            category = elements[2]
            if category != 'Car':
                continue

            sid = self.sequence_num
            oid = int(elements[1])
            file_name = f'km_{sid:02d}{fid:05d}{oid:03d}'
            img_file_name = f'../kitti-mot/training/image_02/{sid:04d}/{fid:06d}.png'
            proj_mat = Calibration(str(Path.joinpath(self.data, f'training/calib/{sid:04d}.txt')))
            patch_file_name = file_name + '_patch.png'
            mask_file_name = file_name + '_mask.png'
            obj = Object3d(' '.join(elements[2:]))
            one_patch = {'patch' : 'patch/' + patch_file_name,
                    'mask' : 'mask/' + mask_file_name,
                    'image_file' : img_file_name,
                    'fl_x' : proj_mat.f_u,
                    'fl_y' : proj_mat.f_v,
                    'cx' : proj_mat.c_u,
                    'cy' : proj_mat.c_v,
                    'cam_tx' : proj_mat.P[0, 3],
                    'cam_ty' : proj_mat.P[1, 3],
                    'cam_tz' : proj_mat.P[2, 3],
                    'xmin': obj.xmin,
                    'xmax': obj.xmax,
                    'ymin': obj.ymin,
                    'ymax': obj.ymax,
                    'height': obj.h,
                    'width': obj.w,
                    'length': obj.l,
                    'obj_x': obj.t[0],
                    'obj_y': obj.t[1],
                    'obj_z': obj.t[2],
                    'yaw': obj.ry,
                    'w': 1242, 
                    'h': 375,
                    }
            instance_list.append(one_patch)
        return instance_list

    def _get_latents(self, instance: Dict, device)->torch.Tensor:
        with torch.no_grad():
            
            image_tensor = get_image_tensor_from_path(self.data / instance['image_file'])
            patch_tensor = image_tensor[int(instance['ymin']):int(instance['ymax']+1),
                                        int(instance['xmin']):int(instance['xmax']+1),
                                        :]
            patch_tensor = self.img_transform(patch_tensor.permute(2,0,1)).permute(1,2,0).to(device)
            return self.model.get_latents(patch_tensor.permute(2,0,1).unsqueeze(0))
        
    def _apply_mask_and_get_latents(self, mask, instance:Dict, device, color_name) -> torch.Tensor:
        with torch.no_grad():
            image_tensor = get_image_tensor_from_path(self.data / instance['image_file'])
            patch_tensor = image_tensor[int(instance['ymin']):int(instance['ymax']+1),
                                            int(instance['xmin']):int(instance['xmax']+1),
                                            :]
            self._save_img(patch_tensor, self.output / f'origin_patch.png')

            mask_tensor = get_image_mask_tensor_from_path(self.data / f"../car_studio/{instance['mask']}")
            color_mask = mask_tensor.detach().clone() * torch.Tensor(mask)[None, None, :]

            patch_tensor[mask_tensor.squeeze(-1), :] = patch_tensor[mask_tensor.squeeze(-1), :] * 0.5 + \
                                                color_mask[mask_tensor.squeeze(-1), :] * 0.5

            self._save_img(patch_tensor, self.output / f'{color_name}_patch.png')
            
            patch_tensor = self.img_transform(patch_tensor.permute(2,0,1)).permute(1,2,0).to(device)
            return self.model.get_latents(patch_tensor.permute(2,0,1).unsqueeze(0)), mask_tensor

        

    def _build_trans(self):
        # rot = self._roty(-np.pi / 2)
        left_trans = np.identity(4)
        left_trans[:3, 3] = np.array([-3, 0., 0.])
        all_views = {
                    # 'left':left_trans,
                    'front':np.identity(4)
                    }
        return all_views
    
    def _generate_all_seq(self, optimized_frames, unoptimized_frames, device):
        for current_frame_id in range(self.start_id, self.end_id + 1):
            doopt_canvas = torch.ones((375, 1242, 3),dtype=torch.float32, device=device)
            noopt_canvas = torch.ones((375, 1242, 3),dtype=torch.float32, device=device)
            all_canvas = torch.ones((375, 1242, 3),dtype=torch.float32, device=device)

            optimized_instances = optimized_frames.get(current_frame_id)
            unoptimized_instances = unoptimized_frames.get(current_frame_id)
            instances = []
            if optimized_instances is not None:
                instances = instances + optimized_instances
            if unoptimized_instances is not None:
                instances = instances + unoptimized_instances
            
            if len(instances) > 0:
                instances = sorted(instances, key=lambda x: x['obj_z'], reverse=True)
                for instance in instances:
                    obj_id = self._get_obj_id(instance)
                    if instance in optimized_instances:
                        latents = self.global_latents_lookup_table[obj_id]
                    else:
                        latents = self._get_latents(instance, device=device)
                    out_img = self._get_output_images(latents=latents, \
                        new_instance=instance, device=device)

                    # self._save_img(out_img['rgb_fine'],
                    #     self.output / f'{obj_id}/{current_frame_id:04d}_patch.png')
                    # self._save_img(out_img['accumulation_fine'].squeeze(-1),
                    #     self.output / f'{obj_id}/{current_frame_id:04d}_mask.png')
                    # self._save_img(out_img['fine_intersect'].to(torch.float).squeeze(-1),
                    #     self.output / f'{obj_id}/{current_frame_id:04d}_intersect.png')
                    canvas_mask = torch.zeros((375, 1242, 3),dtype=torch.bool, device=device)
                    canvas_mask[int(instance['ymin']):int(instance['ymax']+1),
                        int(instance['xmin']):int(instance['xmax']+1), :] =  out_img['accumulation_fine']
                    if instance in optimized_instances:
                        doopt_canvas[canvas_mask] = out_img['rgb_fine'][out_img['accumulation_fine'].squeeze(-1).to(torch.bool),:].flatten()
                    else:
                        noopt_canvas[canvas_mask] = out_img['rgb_fine'][out_img['accumulation_fine'].squeeze(-1).to(torch.bool),:].flatten()
                    all_canvas[canvas_mask] = out_img['rgb_fine'][out_img['accumulation_fine'].squeeze(-1).to(torch.bool),:].flatten()
                self._save_img(doopt_canvas,
                    self.output / f'doopt_{current_frame_id:04d}_front.png')
                self._save_img(noopt_canvas,
                    self.output / f'noopt_{current_frame_id:04d}_front.png')
                self._save_img(all_canvas,
                    self.output / f'all_{current_frame_id:04d}_front.png')
    
    def _get_all_car_rotated(self, optimized_frames,
                                  unoptimized_frames,
                                  device):
        for current_frame_id in self.rotated_frame:
            rot_offsets = np.linspace(0, 2 * np.pi, 20)
            for idx, rot_offset in enumerate(rot_offsets):
                doopt_canvas = torch.ones((375, 1242, 3),dtype=torch.float32, device=device)
                noopt_canvas = torch.ones((375, 1242, 3),dtype=torch.float32, device=device)
                all_canvas = torch.ones((375, 1242, 3),dtype=torch.float32, device=device)

                optimized_instances = copy.deepcopy(optimized_frames.get(current_frame_id))
                unoptimized_instances = copy.deepcopy(unoptimized_frames.get(current_frame_id))
                instances = []
                if optimized_instances is not None:
                    instances = instances + optimized_instances
                if unoptimized_instances is not None:
                    instances = instances + unoptimized_instances

                self._rotate_all_instances_with_xyxy_mask(instances, rot_offset)


                if len(instances) > 0:
                    instances = sorted(instances, key=lambda x: x['obj_z'], reverse=True)
                    for instance in instances:
                        obj_id = self._get_obj_id(instance)
                        if instance in optimized_instances:
                            latents = self.global_latents_lookup_table[obj_id]
                        else:
                            latents = self._get_latents(instance, device=device)
                        out_img = self._get_output_images(latents=latents, \
                            new_instance=instance, device=device)

                        canvas_mask = torch.zeros((375, 1242, 3),dtype=torch.bool, device=device)
                        canvas_mask[int(instance['ymin']):int(instance['ymax']+1),
                            int(instance['xmin']):int(instance['xmax']+1), :] =  out_img['accumulation_fine']
                        if instance in optimized_instances:
                            doopt_canvas[canvas_mask] = out_img['rgb_fine'][out_img['accumulation_fine'].squeeze(-1).to(torch.bool),:].flatten()
                        else:
                            noopt_canvas[canvas_mask] = out_img['rgb_fine'][out_img['accumulation_fine'].squeeze(-1).to(torch.bool),:].flatten()
                        all_canvas[canvas_mask] = out_img['rgb_fine'][out_img['accumulation_fine'].squeeze(-1).to(torch.bool),:].flatten()
                    self._save_img(doopt_canvas,
                        self.output / f'doopt_{current_frame_id:04d}_rot_{idx}.png')
                    self._save_img(noopt_canvas,
                        self.output / f'noopt_{current_frame_id:04d}_rot_{idx}.png')
                    self._save_img(all_canvas,
                        self.output / f'all_{current_frame_id:04d}_rot_{idx}.png')
                    
    def _rotate_all_instances_with_xyxy_mask(self, all_instances, rot_offset) -> None:
        for idx, _ in enumerate(all_instances):
            instance = all_instances[idx]
            tx = instance['obj_x']
            ty = instance['obj_y']
            tz = instance['obj_z']
            l = instance['length']
            h = instance['height']
            w = instance['width']
            imshape_h = instance['h']
            imshape_w = instance['w']
            ry = instance['yaw'] + rot_offset
            P = np.array([[instance['fl_x'], 0, instance['cx'], instance['cam_tx']],
                        [0, instance['fl_y'], instance['cy'], instance['cam_ty']],
                        [0, 0, 1, instance['cam_tz']]])
            corners = self._get_corners([tx, ty, tz, l, h, w, ry])
            pixels = self._project_to_image(corners, P)
            xmin, ymin = np.min(pixels.astype(int), axis=0)
            xmax, ymax = np.max(pixels.astype(int), axis=0)
            xmin, xmax = np.clip(np.array([xmin, xmax]),
                                 0, imshape_w - 1)
            ymin, ymax = np.clip(np.array([ymin, ymax]),
                                 0, imshape_h - 1)
            instance['xmin'] = xmin
            instance['xmax'] = xmax
            instance['ymin'] = ymin
            instance['ymax'] = ymax
            instance['yaw'] = self._normalize_rot_yaw(ry)

    def _normalize_rot_yaw(self, yaw):
        rot = (yaw + 2 * np.pi) % (2 * np.pi)
        if rot > np.pi:
            rot = rot - 2 * np.pi
        return rot

    def _get_one_car_translation(self, optimized_frames,
                                  unoptimized_frames, device):
        
        for current_frame_id in self.trans_frame:
            x_offsets = np.linspace(0, 2.5, 20, endpoint=False)
            z_offsets = np.linspace(0, -25.0, 60)

            pos_offsets = np.zeros((80,3), dtype=float)
            pos_offsets[:20, 0] = x_offsets
            pos_offsets[20:, 0] = 2.5
            pos_offsets[20:, 2] = z_offsets

            for idx, pos_offset in enumerate(pos_offsets):
                doopt_canvas = torch.ones((375, 1242, 3),dtype=torch.float32, device=device)
                noopt_canvas = torch.ones((375, 1242, 3),dtype=torch.float32, device=device)
                all_canvas = torch.ones((375, 1242, 3),dtype=torch.float32, device=device)

                optimized_instances = copy.deepcopy(optimized_frames.get(current_frame_id))
                unoptimized_instances = copy.deepcopy(unoptimized_frames.get(current_frame_id))
                instances = []
                if optimized_instances is not None:
                    instances = instances + optimized_instances
                if unoptimized_instances is not None:
                    instances = instances + unoptimized_instances

                self._trans_all_instances_with_xyxy_mask(instances, pos_offset)


                if len(instances) > 0:
                    instances = sorted(instances, key=lambda x: x['obj_z'], reverse=True)
                    for instance in instances:
                        obj_id = self._get_obj_id(instance)
                        if instance in optimized_instances:
                            latents = self.global_latents_lookup_table[obj_id]
                        else:
                            latents = self._get_latents(instance, device=device)
                        out_img = self._get_output_images(latents=latents, \
                            new_instance=instance, device=device)

                        canvas_mask = torch.zeros((375, 1242, 3),dtype=torch.bool, device=device)
                        canvas_mask[int(instance['ymin']):int(instance['ymax']+1),
                            int(instance['xmin']):int(instance['xmax']+1), :] =  out_img['accumulation_fine']
                        if instance in optimized_instances:
                            doopt_canvas[canvas_mask] = out_img['rgb_fine'][out_img['accumulation_fine'].squeeze(-1).to(torch.bool),:].flatten()
                        else:
                            noopt_canvas[canvas_mask] = out_img['rgb_fine'][out_img['accumulation_fine'].squeeze(-1).to(torch.bool),:].flatten()
                        all_canvas[canvas_mask] = out_img['rgb_fine'][out_img['accumulation_fine'].squeeze(-1).to(torch.bool),:].flatten()
                    self._save_img(doopt_canvas,
                        self.output / f'doopt_{current_frame_id:04d}_trans_{idx}.png')
                    self._save_img(noopt_canvas,
                        self.output / f'noopt_{current_frame_id:04d}_trans_{idx}.png')
                    self._save_img(all_canvas,
                        self.output / f'all_{current_frame_id:04d}_trans_{idx}.png')

    def _trans_all_instances_with_xyxy_mask(self, all_instances, translation) -> None:
        for idx, _ in enumerate(all_instances):
            instance = all_instances[idx]
            obj_id = self._get_obj_id(instance)
            if obj_id in self.trans_instance:
                tx = instance['obj_x'] + translation[0]
                ty = instance['obj_y'] + translation[1]
                tz = instance['obj_z'] + translation[2]
                l = instance['length']
                h = instance['height']
                w = instance['width']
                imshape_h = instance['h']
                imshape_w = instance['w']
                ry = instance['yaw']
                P = np.array([[instance['fl_x'], 0, instance['cx'], instance['cam_tx']],
                            [0, instance['fl_y'], instance['cy'], instance['cam_ty']],
                            [0, 0, 1, instance['cam_tz']]])
                corners = self._get_corners([tx, ty, tz, l, h, w, ry])
                pixels = self._project_to_image(corners, P)
                xmin, ymin = np.min(pixels.astype(int), axis=0)
                xmax, ymax = np.max(pixels.astype(int), axis=0)
                xmin, xmax = np.clip(np.array([xmin, xmax]),
                                     0, imshape_w - 1)
                ymin, ymax = np.clip(np.array([ymin, ymax]),
                                     0, imshape_h - 1)
                instance['obj_x'] = tx
                instance['obj_y'] = ty
                instance['obj_z'] = tz
                instance['xmin'] = xmin
                instance['xmax'] = xmax
                instance['ymin'] = ymin
                instance['ymax'] = ymax
    
    def _change_colors(self, optimized_frames,
                            unoptimized_frames,
                            device):
        color_scheme = {
            'aque': np.array([0, 255, 255], dtype=float) / 255.0,
            'white': np.array([255, 255, 255], dtype=float) / 255.0,
            'blue': np.array([0, 0, 255], dtype=float) / 255.0,
            'violet': np.array([138, 43, 226], dtype=float) / 255.0,
            'brown': np.array([255, 64, 64], dtype=float) / 255.0,
            'coral': np.array([255, 127, 80], dtype=float) / 255.0,
            'cyan': np.array([0, 238, 238], dtype=float) / 255.0,
            'black': np.array([0, 0, 0], dtype=float) / 255.0,
            'emerald': np.array([0, 201, 87], dtype=float) / 255.0,
            'green': np.array([0, 255, 0], dtype=float) / 255.0,
            'orange': np.array([255, 128, 0], dtype=float) / 255.0,
        }
        for current_frame_id in self.colors_frame:
            for color_name, color_mask in color_scheme.items():
                doopt_canvas = torch.ones((375, 1242, 3),dtype=torch.float32, device=device)
                noopt_canvas = torch.ones((375, 1242, 3),dtype=torch.float32, device=device)
                all_canvas = torch.ones((375, 1242, 3),dtype=torch.float32, device=device)

                optimized_instances = copy.deepcopy(optimized_frames.get(current_frame_id))
                unoptimized_instances = copy.deepcopy(unoptimized_frames.get(current_frame_id))
                instances = []
                if optimized_instances is not None:
                    instances = instances + optimized_instances
                if unoptimized_instances is not None:
                    instances = instances + unoptimized_instances

                if len(instances) > 0:
                    instances = sorted(instances, key=lambda x: x['obj_z'], reverse=True)
                    for instance in instances:
                        obj_id = self._get_obj_id(instance)
                        if obj_id in self.color_instance:
                            latents, mask = self._apply_mask_and_get_latents(color_mask,
                                                                       instance,
                                                                       device,
                                                                       color_name)
                        elif (optimized_instances is not None) and (instance in optimized_instances):
                            latents = self.global_latents_lookup_table[obj_id]
                        else:
                            latents = self._get_latents(instance, device=device)
                        out_img = self._get_output_images(latents=latents, \
                            new_instance=instance, device=device)
                        
                        if obj_id in self.color_instance:
                            out_img['accumulation_fine'] = torch.Tensor(mask).to(torch.float).to(device)
                            self._save_img(out_img['rgb_fine'], self.output / f'{color_name}_recon_patch.png')
                            del mask

                        canvas_mask = torch.zeros((375, 1242, 3),dtype=torch.bool, device=device)
                        canvas_mask[int(instance['ymin']):int(instance['ymax']+1),
                            int(instance['xmin']):int(instance['xmax']+1), :] =  out_img['accumulation_fine']
                        if optimized_instances is not None and instance in optimized_instances:
                            doopt_canvas[canvas_mask] = out_img['rgb_fine'][out_img['accumulation_fine'].squeeze(-1).to(torch.bool),:].flatten()
                        else:
                            noopt_canvas[canvas_mask] = out_img['rgb_fine'][out_img['accumulation_fine'].squeeze(-1).to(torch.bool),:].flatten()
                        all_canvas[canvas_mask] = out_img['rgb_fine'][out_img['accumulation_fine'].squeeze(-1).to(torch.bool),:].flatten()
                    self._save_img(doopt_canvas,
                        self.output / f'doopt_{current_frame_id:04d}_color_{color_name}.png')
                    self._save_img(noopt_canvas,
                        self.output / f'noopt_{current_frame_id:04d}_color_{color_name}.png')
                    self._save_img(all_canvas,
                        self.output / f'all_{current_frame_id:04d}_color_{color_name}.png')


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderNovelView).main()


if __name__ == "__main__":
    entrypoint()
