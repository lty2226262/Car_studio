# 2023 Tianyu LIU [tliubk@connect.ust.hk] Copyright. All rights reserved.
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

""" Data parser for carstudio datasets. """

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Type

import numpy as np
import torch
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import (CAMERA_MODEL_TO_TYPE, Cameras,
                                        CameraType)
from nerfstudio.data.dataparsers.base_dataparser import (DataParser,
                                                         DataParserConfig,
                                                         DataparserOutputs)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image

MAX_AUTO_RESOLUTION = 1600


@dataclass
class CarstudioDataParserConfig(DataParserConfig):
    """Carstudio dataset config"""

    _target: Type = field(default_factory=lambda: Carstudio)
    """target class to instantiate"""
    data: Path = Path('./data/car_studio/')
    """Directory specifying location of data."""
    subset: Tuple[Literal["vk", "ko", "ns", "km", "dv"], ...] = (
        "ko",
    )
    """which dataset subset to use, options: vk(vkitti), ko(kittiobj), ns(nuscenes), km(kittimot)"""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""



@dataclass
class Carstudio(DataParser):
    """Carstudio DatasetParser"""

    config: CarstudioDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements

        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        if isinstance(self.config.subset, str):
            self.config.subset = tuple([self.config.subset])

        meta_list = []
        for sub_set in self.config.subset:
            meta = load_from_json(self.config.data / (sub_set + ".json"))
            meta_list.append(meta)

        patch_filenames = []
        mask_filenames = []
        depth_filenames = []
        xyxy_masks = []
        poses = []
        num_skipped_image_filenames = 0
        src_img_filenames = []
        object_ids = []

        fx_fixed = "fl_x" in meta_list
        fy_fixed = "fl_y" in meta_list
        cx_fixed = "cx" in meta_list
        cy_fixed = "cy" in meta_list
        height_fixed = "h" in meta_list
        width_fixed = "w" in meta_list

        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta_list:
                distort_fixed = True
                break
        
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for meta_all in meta_list:
            CONSOLE.print(f"Begin to process {meta_all['dataset_name']} {split}")
            
            for instance_it in meta_all["instances"]:
                patch_filepath = Path(instance_it["patch"])
                data_dir = self.config.data
                patch_fname = self._get_fname(patch_filepath, data_dir)
                if not patch_fname.exists():
                    num_skipped_image_filenames += 1
                    continue

                if not fx_fixed:
                    assert "fl_x" in instance_it, "fx not specified in frame"
                    fx.append(float(instance_it["fl_x"]))
                if not fy_fixed:
                    assert "fl_y" in instance_it, "fy not specified in frame"
                    fy.append(float(instance_it["fl_y"]))
                if not cx_fixed:
                    assert "cx" in instance_it, "cx not specified in frame"
                    cx.append(float(instance_it["cx"]))
                if not cy_fixed:
                    assert "cy" in instance_it, "cy not specified in frame"
                    cy.append(float(instance_it["cy"]))
                if not height_fixed:
                    assert "h" in instance_it, "height not specified in frame"
                    height.append(int(instance_it["h"]))
                if not width_fixed:
                    assert "w" in instance_it, "width not specified in frame"
                    width.append(int(instance_it["w"]))
                    
                if not distort_fixed:
                    distort.append(
                        camera_utils.get_distortion_params(
                            k1=float(instance_it["k1"]) if "k1" in instance_it else 0.0,
                            k2=float(instance_it["k2"]) if "k2" in instance_it else 0.0,
                            k3=float(instance_it["k3"]) if "k3" in instance_it else 0.0,
                            k4=float(instance_it["k4"]) if "k4" in instance_it else 0.0,
                            p1=float(instance_it["p1"]) if "p1" in instance_it else 0.0,
                            p2=float(instance_it["p2"]) if "p2" in instance_it else 0.0,
                        )
                    )

                patch_filenames.append(patch_fname)
                xyxy = np.array([instance_it['xmin'],
                                 instance_it['ymin'],
                                 instance_it['xmax'],
                                 instance_it['ymax'],])
                xyxy_masks.append(xyxy)

                pose = self._get_pose(instance_it)
                poses.append(pose)

                object_id = self._get_obj_id(instance_it, meta_all['dataset_name'])
                object_ids.append(object_id)
                # camera = Cameras(
                #     fx=fx[-1],
                #     fy=fy[-1],
                #     cx=cx[-1],
                #     cy=cy[-1],
                #     distortion_params=distort[-1],
                #     height=height[-1],
                #     width=width[-1],
                #     camera_to_worlds=torch.from_numpy(pose[None, :3, :4]),
                #     camera_type=CameraType.PERSPECTIVE,)
                # scene_box = SceneBox(torch.tensor([[-1., -1., -1.],
                #                        [1.0, 1.0, 1.0]]))
                # rays = camera.generate_rays(0, 
                #                             camera_opt_to_camera = torch.tensor([[1, 0, 0, 0],
                #                                                                 [0, 1, 0, 0],
                #                                                                 [0, 0, 1, 0]], dtype=torch.float64),
                #                             aabb_box = scene_box,
                #                             disable_distortion=True               
                #                             )
                
                # objs_pose = np.array([instance_it['obj_x'], instance_it['obj_y'], instance_it['obj_z']])
                # objs_dim = np.array([instance_it['length'], instance_it['height'], instance_it['width']])
                # objs_yaw = np.array([instance_it['yaw']])
                            
                # objs = np.concatenate([objs_pose, objs_dim, objs_yaw], -1).reshape(1, 7)
       
                # cam_pos = torch.eye(4)[None, :, :]
                # cam_pos[:, 2, 2] = -1
                # cam_pos[:, 1, 1] = -1

                # render_rays = self.gen_rays(
                #             cam_pos, width[-1], height[-1], 
                #             torch.tensor([fx[-1], fy[-1]]), 0, np.inf,
                #             torch.tensor([cx[-1], cy[-1]])
                #         )[0].flatten(0,1).numpy()               
                # ray_o = self.world2object(np.zeros((len(render_rays), 3)), objs)
                # ray_d = self.world2object(render_rays[:, 3:6], objs, use_dir=True)

                # z_in, z_out, intersect = self.ray_box_intersection(ray_o, ray_d)
                # bounds =  np.ones((*ray_o.shape[:-1], 2)) * -1
                # bounds [intersect, 0] = z_in
                # bounds [intersect, 1] = z_out

                if "mask" in instance_it:
                    mask_filepath = Path(instance_it["mask"])
                    mask_fname = self._get_fname(
                        mask_filepath,
                        data_dir,
                        downsample_folder_prefix="masks_",
                    )
                    mask_filenames.append(mask_fname)

                if "depth_file_path" in instance_it:
                    depth_filepath = Path(instance_it["depth_file_path"])
                    depth_fname = self._get_fname(depth_filepath, data_dir, downsample_folder_prefix="depths_")
                    depth_filenames.append(depth_fname)
                
                if "image_file" in instance_it:
                    image_filepath = Path(instance_it["image_file"])
                    image_fname = self._get_fname(image_filepath, data_dir, downsample_folder_prefix='src_images')
                    src_img_filenames.append(image_fname)

        if num_skipped_image_filenames >= 0:
            CONSOLE.log(f"Skipping {num_skipped_image_filenames} files in dataset split {split}.")
        assert (
            len(patch_filenames) != 0
        ), """
        No image files found. 
        You should check the file_paths in the {dataset}.json file to make sure they are correct.
        """
        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(patch_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in {dataset}.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(patch_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in {dataset}.json.
        """

        assert len(src_img_filenames) == 0 or (
            len(src_img_filenames) == len(patch_filenames)
        ), """
        Different number of image and src_image filenames.
        You should check that src_img_filenames is specified for every frame (or zero frames) in {dataset}.json.
        """
        assert len(xyxy_masks) == 0 or (
            len(xyxy_masks) == len(patch_filenames)
        ), """
        Different number of image and xyxy_masks.
        You should check that xyxy_masks is specified for every frame (or zero frames) in {dataset}.json.
        """
        assert len(object_ids) == 0 or (
            len(object_ids) == len(patch_filenames)
        ), """
        Different number of image and object_ids.
        You should check that object_ids is specified for every frame (or zero frames) in {dataset}.json.
        """

        has_split_files_spec = any(f"{split}_filenames" in meta for split in ("train", "val", "test"))
        if f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(self._get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"])
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(f"Some filenames for split {split} were not found: {unmatched_filenames}.")

            indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        else:
            # filter image_filenames and poses based on train/eval split percentage
            num_images = len(patch_filenames)
            num_train_images = math.ceil(num_images * self.config.train_split_fraction)
            num_eval_images = num_images - num_train_images
            i_all = np.arange(num_images)
            i_train = np.linspace(
                0, num_images - 1, num_train_images, dtype=int
            )  # equally spaced training images starting and ending at 0 and num_images-1
            i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
            assert len(i_eval) == num_eval_images
            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        # poses_after, transform_matrix = camera_utils.auto_orient_and_center_poses(
        #     poses,
        #     method=orientation_method,
        #     center_method=self.config.center_method,
        # )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        patch_filenames = [patch_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        src_img_filenames = [src_img_filenames[i] for i in indices] if len(src_img_filenames) > 0 else []

        object_ids = [object_ids[i] for i in indices] if len(object_ids) > 0 else [] 

        xyxy_masks = torch.from_numpy(np.array(xyxy_masks).astype(np.float32))

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]
        xyxy_masks = xyxy_masks[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        dataparser_outputs = DataparserOutputs(
            image_filenames=patch_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "src_image_files": src_img_filenames if len(src_img_filenames) > 0 else None,
                "xyxy_masks": xyxy_masks if len(xyxy_masks) > 0 else None,
                "object_ids": object_ids if len(object_ids) > 0 else None,
                "split": split,
            },
        )
        return dataparser_outputs
    
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
    
    def _get_obj_id(self, instance_it: Dict, dataset_name: str)-> str:
        """get object if from the file name"""
        name_parser_map = {
            'km': {
                'sid':[9,11],
                'oid':[16,19],
            },
            'vk': {
                'sid':[9,11],
                'oid':[16,19],
            },
            'ko': {
                'sid':[9,11],
                'oid':[16,21],
            },
            'ns': {
                'sid':[9,12],
                'oid':[17,22],
            },
            'dvm_car_studio': {
                'sid':[9, 10],
                'oid':[16, 22]
            }
        }
        sid_range = name_parser_map[dataset_name]['sid']
        sid = instance_it['patch'][sid_range[0]:sid_range[1]]
        oid_range = name_parser_map[dataset_name]['oid']
        oid = instance_it['patch'][oid_range[0]:oid_range[1]]
        return f'{dataset_name}{sid}{oid}'
        
    
    # def _box3d_to_image_roi(self, instance_it, imshape=None):
    #     '''convert box3d to image roi'''
    #     obj = [
    #         instance_it['obj_x'],
    #         instance_it['obj_y'],
    #         instance_it['obj_z'],
    #         instance_it['length'],
    #         instance_it['height'],
    #         instance_it['width'],
    #         instance_it['yaw']
    #     ]
    #     corners_3d = self._get_corners(obj)
    #     P = np.array([[instance_it['fl_x'], 0, instance_it['cx'], instance_it['cam_tx']],
    #                    [0, instance_it['fl_y'], instance_it['cy'], instance_it['cam_ty']],
    #                    [0, 0, 1, instance_it['cam_tz']]])

    #     # project the 3d bounding box into the image plane
    #     corners_2d = self._project_to_image(corners_3d, P)
    #     xmin, ymin = np.min(corners_2d, axis=0)
    #     xmax, ymax = np.max(corners_2d, axis=0)

    #     if imshape is not None:
    #         xmin = np.clip(xmin, 0, imshape[1])
    #         xmax = np.clip(xmax, 0, imshape[1])
    #         ymin = np.clip(ymin, 0, imshape[0])
    #         ymax = np.clip(ymax, 0, imshape[0])

    #     return (xmin, ymin, xmax, ymax)
    

    def _get_fname(self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return data_dir / filepath

    def _get_corners(self, obj):
        if isinstance(obj, list):
            tx, ty, tz, l, h, w, ry = obj
        else:
            tx, ty, tz, l, h, w, ry = obj.tolist()
        
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

    def object2world(self, pts, objs):
        # pts: Nb x Np x 3
        # objs: Nb x 7

        pose, dim, theta_y = objs[:, :3], objs[:, 3:6], objs[:, 6:]

        pose_w = pose.clone()
        pose_w[:, 1] -= dim[:, 1] / 2

        N_pts = pts.shape[1]

        pose_w = pose_w[:, None, :].repeat(1, N_pts, 1)
        theta_y = theta_y[:, None, :].repeat(1, N_pts, 1)
        dim = dim[:, None, :].repeat(1, N_pts, 1)

        pts_w = pts * (dim / 2 + 1e-9)
        t_w_o = self.rotate_yaw_torch(-pose_w, theta_y)
        pts_w = self.rotate_yaw_torch(pts_w - t_w_o, -theta_y) 

        return pts_w

    def rotate_yaw_torch(self, p, yaw):
        """Rotates p with yaw in the given coord frame with y being the relevant axis and pointing downwards
        Args:
            p: 3D points in a given frame [N_pts, N_frames, 3]/[N_pts, N_frames, N_samples, 3]
            yaw: Rotation angle
        Returns:
            p: Rotated points [N_pts, N_frames, N_samples, 3]
        """
        c_y = torch.cos(yaw)
        s_y = torch.sin(yaw)

        p_x = c_y * p[..., [0]] - s_y * p[..., [2]]
        p_y = p[..., [1]]
        p_z = s_y * p[..., [0]] + c_y * p[..., [2]]

        return torch.cat([p_x, p_y, p_z], dim=-1)

    def gen_rays(self, poses, width, height, focal, z_near, z_far, c=None, ndc=False):
        """
        Generate camera rays
        :return (B, H, W, 8)
        """
        num_images = poses.shape[0]
        device = poses.device
        cam_unproj_map = (
            self.unproj_map(width, height, focal.squeeze(), c=c, device=device)
            .unsqueeze(0)
            .repeat(num_images, 1, 1, 1)
        )
        cam_centers = poses[:, None, None, :3, 3].expand(-1, height, width, -1)
        cam_raydir = torch.matmul(
            poses[:, None, None, :3, :3], cam_unproj_map.unsqueeze(-1)
        )[:, :, :, :, 0]
        

        cam_nears = (
            torch.tensor(z_near, device=device)
            .view(1, 1, 1, 1)
            .expand(num_images, height, width, -1)
        )
        cam_fars = (
            torch.tensor(z_far, device=device)
            .view(1, 1, 1, 1)
            .expand(num_images, height, width, -1)
        )
        return torch.cat(
            (cam_centers, cam_raydir, cam_nears, cam_fars), dim=-1
        )  # (B, H, W, 8)

    def unproj_map(self, width, height, f, c=None, device="cpu"):
        """
        Get camera unprojection map for given image size.
        [y,x] of output tensor will contain unit vector of camera ray of that pixel.
        :param width image width
        :param height image height
        :param f focal length, either a number or tensor [fx, fy]
        :param c principal point, optional, either None or tensor [fx, fy]
        if not specified uses center of image
        :return unproj map (height, width, 3)
        """
        if c is None:
            c = [width * 0.5, height * 0.5]
        else:
            c = c.squeeze()
        if isinstance(f, float):
            f = [f, f]
        elif len(f.shape) == 0:
            f = f[None].expand(2)
        elif len(f.shape) == 1:
            f = f.expand(2)
        Y, X = torch.meshgrid(
            torch.arange(height, dtype=torch.float32) - float(c[1]),
            torch.arange(width, dtype=torch.float32) - float(c[0]),
        )
        X = X.to(device=device) / float(f[0])
        Y = Y.to(device=device) / float(f[1])
        Z = torch.ones_like(X)
        unproj = torch.stack((X, -Y, -Z), dim=-1)
        unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
        return unproj
    def world2object(self, pts, obj, use_dir=False):
    
        pose, dim, theta_y = obj[:, :3], obj[:, 3:6], obj[:, 6:]
        
        pose_w = pose.copy()
        pose_w[:, 1] -= dim[:, 1] / 2

        N_obj = pose.shape[0]
        N_pts = pts.shape[0]

        pose_w = np.repeat(pose_w[None, :, :], N_pts, axis=0)
        theta_y = np.repeat(theta_y[None, :, :], N_pts, axis=0)
        dim = np.repeat(dim[None, :, :], N_pts, axis=0)

        # Describes the origin of the world system w in the object system o
        t_w_o = self.rotate_yaw(-pose_w, theta_y)
        N_obj = theta_y.shape[1]

        pts_w = np.repeat(pts[:, np.newaxis, ...], N_obj, axis=1)
    
        if use_dir:
            # for dir, no need to shift
            pts_o = self.rotate_yaw(pts_w, theta_y)
        else:
            pts_o = self.rotate_yaw(pts_w, theta_y) + t_w_o
    
        # Scale rays_o_v and rays_d_v for box [[-1.,1], [-1.,1], [-1.,1]]
        pts_o = pts_o / (dim / 2 + 1e-9)
        
        if use_dir:
            pts_o = pts_o / np.linalg.norm(pts_o, axis=-1, keepdims=True)
    
        return pts_o
    

    def rotate_yaw(self, p, yaw):
        """Rotates p with yaw in the given coord frame with y being the relevant axis and pointing downwards
        Args:
            p: 3D points in a given frame [N_pts, N_frames, 3]/[N_pts, N_frames, N_samples, 3]
            yaw: Rotation angle
        Returns:
            p: Rotated points [N_pts, N_frames, N_samples, 3]
        """
        c_y = np.cos(yaw)
        s_y = np.sin(yaw)

        p_x = c_y * p[..., [0]] - s_y * p[..., [2]]
        p_y = p[..., [1]]
        p_z = s_y * p[..., [0]] + c_y * p[..., [2]]

        return np.concatenate([p_x, p_y, p_z], axis=-1)

    def ray_box_intersection(self, ray_o, ray_d, aabb_min=None, aabb_max=None):
        """Returns 1-D intersection point along each ray if a ray-box intersection is detected
        If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary
        Args:
            ray_o: Origin of the ray in each box frame, [rays, boxes, 3]
            ray_d: Unit direction of each ray in each box frame, [rays, boxes, 3]
            (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
            (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified
        Returns:
            z_ray_in:
            z_ray_out:
            intersection_map: Maps intersection values in z to their ray-box intersection
        """
        # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
        # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
        if aabb_min is None:
            aabb_min = np.ones_like(ray_o) * -1. # tf.constant([-1., -1., -1.])
        if aabb_max is None:
            aabb_max = np.ones_like(ray_o) # tf.constant([1., 1., 1.])

        inv_d = np.reciprocal(ray_d)

        t_min = (aabb_min - ray_o) * inv_d
        t_max = (aabb_max - ray_o) * inv_d

        t0 = np.minimum(t_min, t_max)
        t1 = np.maximum(t_min, t_max)

        t_near = np.maximum(np.maximum(t0[..., 0], t0[..., 1]), t0[..., 2])
        t_far = np.minimum(np.minimum(t1[..., 0], t1[..., 1]), t1[..., 2])

        # Check if rays are inside boxes
        intersection_map = t_far > t_near # np.where(t_far > t_near)[0]
        
        # Check that boxes are in front of the ray origin
        positive_far = (t_far * intersection_map) > 0
        intersection_map = np.logical_and(intersection_map, positive_far)

        if not intersection_map.shape[0] == 0:
            z_ray_in = t_near[intersection_map]
            z_ray_out = t_far[intersection_map]
        else:
            return None, None, None

        return z_ray_in, z_ray_out, intersection_map

CarstudioDataParserConfigSpecification = DataParserSpecification(config=CarstudioDataParserConfig)
