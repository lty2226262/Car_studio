# Ref from: https://github.com/kuixu/kitti_object_vis/blob/master/kitti_util.py
# Modified by Tianyu LIU [tliubk@connect.ust.hk]. All rights reserved.
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
from typing import Dict, List, Literal, Tuple

import imageio
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


class Calibration():
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''
    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)

        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3,4])
        
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3,4])
        self.C2V = inverse_rigid_trans(self.V2C)
        
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0,[3,3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0,2]
        self.c_v = self.P[1,2]
        self.f_u = self.P[0,0]
        self.f_v = self.P[1,1]
        self.b_x = self.P[0,3]/(-self.f_u) # relative 
        self.b_y = self.P[1,3]/(-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0:
                    continue
                if ':' in line:
                    key, value = line.split(':', 1)
                else:
                    key, value = line.split(' ', maxsplit=1)
                    if key == 'Tr_velo_cam':
                        key = 'Tr_velo_to_cam'
                    elif key == 'R_rect':
                        key = 'R0_rect'
                    elif key == 'Tr_imu_velo': 
                        key = 'Tr_imu_to_velo'
                    else: 
                        raise ValueError('invalid key value')
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data



class Object3d(object):
    ''' 3d object label '''
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0] # 'Car', 'Pedestrian', ...
        self.truncation = data[1] # truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4] # left
        self.ymin = data[5] # top
        self.xmax = data[6] # right
        self.ymax = data[7] # bottom
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        
        # extract 3d bounding box information
        self.h = data[8] # box height
        self.w = data[9] # box width
        self.l = data[10] # box length (in meters)
        self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
        self.dim = (self.l, self.h, self.w)
        self.ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]


@dataclass
class ProcessVKitti():
    ''' Parallel processing v-kitti dataset
    '''
    dataset_dir: Path
    """Path to dataset root"""
    dataset_type: Literal
    """Dataset type, must be 'vk' """
    car_type: Tuple[Literal['Car', 'Truck', 'Van'], ...] = ("Car")
    """Car type to process, 'Car', 'Truck' and 'Van' are valid"""


    def __post_init__(self)->None:
        assert self.dataset_type == "vk"
        self.seqs = [i for i in os.listdir(self.dataset_dir)
                     if os.path.isdir(os.path.join(self.dataset_dir, i)) 
                     and 'Scene' in i]
        self.car_instances = list()
        
        for seq in self.seqs:
            frame_dict, oid_dict = read_vkitti_tracking_label(
                Path.joinpath(self.dataset_dir, seq, 'clone/'), self.car_type)
            self.car_instances += generate_file_path_list_from_obj_dict(frame_dict, oid_dict, seq)
        
        sam_checkpoint = "dependencies/segment-anything/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)

    def __len__(self)->int:
        return len(self.car_instances)

    def __getitem__(self, idx)->Tuple[List, List, List]:
        frame = self.car_instances[idx]
        sid = frame['sid']
        fid = frame['fid']
        image_path = Path.joinpath(self.dataset_dir,
                                   f'Scene{sid:02d}/clone/frames/rgb/Camera_0/rgb_{fid:05d}.jpg')
        image = imageio.v2.imread(str(image_path))
        self.predictor.set_image(image)

        input_boxes = list()
        idx_helper = list()

        for idx, car_obj in enumerate(frame['car_objs']):
            input_box = car_obj.box2d
            xmin, ymin, xmax, ymax = input_box
            # filter out small patches
            if (ymax - ymin) < 64 or (xmax - xmin) < 32 or car_obj.t[2] > 50.0:
                pass
            else:
                input_boxes.append(input_box)
                idx_helper.append(idx)

        patches = list()
        masked_patches = list()
        json_segment = list()
        if len(input_boxes) == 0:
            return (patches, masked_patches, json_segment)
        else:
            input_boxes = np.array(input_boxes)

            transformed_boxes = self.predictor.transform.apply_boxes(input_boxes, image.shape[:2])
            transformed_boxes = torch.Tensor(transformed_boxes).to(self.predictor.device)

            masks, ious, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            for idx, iou in enumerate(ious):
                # filter out the patches that is heavily occluded and low confidence
                this_obj = frame['car_objs'][idx_helper[idx]]
                obj_xmin, obj_ymin, obj_xmax, obj_ymax = np.array(this_obj.box2d, dtype=int)
                intersec = masks[idx].sum() / ((obj_xmax+1 - obj_xmin) * (obj_ymax + 1 - obj_ymin))
                if float(iou) > 0.6 and intersec.item() > 0.6:
                    fid = frame['fid']
                    sid = frame['sid']
                    oid = frame['oid'][idx_helper[idx]]
                    mask_to_save = masks[idx, :, obj_ymin: obj_ymax+1, 
                                         obj_xmin: obj_xmax+1].permute((1,2,0)).cpu().numpy()
                    patch_to_save = image[obj_ymin: obj_ymax+1, obj_xmin: obj_xmax+1, :]
                    patches.append(patch_to_save)
                    masked_patches.append(mask_to_save)
                    info = self.generate_vkitti_dict(this_obj, sid, fid, oid)
                    json_segment.append(info)
        
        return patches, masked_patches, json_segment
                    

    def generate_vkitti_dict(self, obj: Object3d, sid: int, fid: int, oid: int):
        '''generate the vkitti info json
        '''
        proj_mat = VkittiCalibration()
        file_name = f'{self.dataset_type}_{sid:02d}{fid:05d}{oid:03d}'
        img_file_name = f'../vkitti/Scene{sid:02d}/clone/frames/rgb/Camera_0/rgb_{fid:05d}.jpg'
        patch_file_name = file_name + '_patch.png'
        mask_file_name = file_name + '_mask.png'
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
                          }
        return one_patch
        
@dataclass
class ProcessKittiMot():
    ''' Parallel processing v-kitti dataset
    '''
    dataset_dir: Path
    """Path to dataset root"""
    dataset_type: Literal
    """Dataset type, must be 'km' """
    car_type: Tuple[Literal['Car', 'Truck', 'Van'], ...] = ("Car")
    """Car type to process, 'Car', 'Truck' and 'Van' are valid"""


    def __post_init__(self)->None:
        assert self.dataset_type == 'km'
        seq_train_image_02_path = self.dataset_dir.joinpath('training/image_02/')
        self.seqs = [i for i in os.listdir(seq_train_image_02_path)
                     if os.path.isdir(os.path.join(self.dataset_dir, 'training/image_02/', i))]
        
        self.car_instances = list()
        
        for seq in self.seqs:
            frame_dict, oid_dict = read_kitti_mot_label(
                Path.joinpath(self.dataset_dir, f'training/label_02/{seq}.txt'), self.car_type)
            self.car_instances += generate_file_path_list_from_obj_dict(frame_dict, oid_dict, seq)
        
        sam_checkpoint = "dependencies/segment-anything/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)

    def __len__(self)->int:
        return len(self.car_instances)

    def __getitem__(self, idx)->Tuple[List, List, List]:
        frame = self.car_instances[idx]
        sid = frame['sid']
        fid = frame['fid']
        image_path = Path.joinpath(self.dataset_dir,
                                   f'training/image_02/{sid:04d}/{fid:06d}.png')
        image = imageio.v2.imread(str(image_path))
        self.predictor.set_image(image)

        input_boxes = list()
        idx_helper = list()

        for idx, car_obj in enumerate(frame['car_objs']):
            input_box = car_obj.box2d
            xmin, ymin, xmax, ymax = input_box
            # filter out small patches
            if (ymax - ymin) < 64 or (xmax - xmin) < 32 or car_obj.t[2] > 50.0:
                pass
            else:
                input_boxes.append(input_box)
                idx_helper.append(idx)

        patches = list()
        masked_patches = list()
        json_segment = list()
        if len(input_boxes) == 0:
            return (patches, masked_patches, json_segment)
        else:
            input_boxes = np.array(input_boxes)

            transformed_boxes = self.predictor.transform.apply_boxes(input_boxes, image.shape[:2])
            transformed_boxes = torch.Tensor(transformed_boxes).to(self.predictor.device)

            masks, ious, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            for idx, iou in enumerate(ious):
                # filter out the patches that is heavily occluded and low confidence
                this_obj = frame['car_objs'][idx_helper[idx]]
                obj_xmin, obj_ymin, obj_xmax, obj_ymax = np.array(this_obj.box2d, dtype=int)
                intersec = masks[idx].sum() / ((obj_xmax+1 - obj_xmin) * (obj_ymax + 1 - obj_ymin))
                if float(iou) > 0.6 and intersec.item() > 0.6:
                    fid = frame['fid']
                    sid = frame['sid']
                    oid = frame['oid'][idx_helper[idx]]
                    mask_to_save = masks[idx, :, obj_ymin: obj_ymax+1, 
                                         obj_xmin: obj_xmax+1].permute((1,2,0)).cpu().numpy()
                    patch_to_save = image[obj_ymin: obj_ymax+1, obj_xmin: obj_xmax+1, :]
                    patches.append(patch_to_save)
                    masked_patches.append(mask_to_save)
                    info = self.generate_kitti_mot_dict(this_obj, sid, fid, oid)
                    json_segment.append(info)
        
        return patches, masked_patches, json_segment
    
    def generate_kitti_mot_dict(self, obj: Object3d, sid: int, fid: int, oid: int):
        '''generate the vkitti info json
        '''
        proj_mat = Calibration(str(Path.joinpath(self.dataset_dir, f'training/calib/{sid:04d}.txt')))
        file_name = f'{self.dataset_type}_{sid:02d}{fid:05d}{oid:03d}'
        img_file_name = f'../kitti-mot/training/image_02/{sid:04d}/{fid:06d}.png'
        patch_file_name = file_name + '_patch.png'
        mask_file_name = file_name + '_mask.png'
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
                          }
        return one_patch


       
@dataclass
class ProcessKittiObj():
    ''' Parallel processing v-kitti dataset
    '''
    dataset_dir: Path
    """Path to dataset root"""
    dataset_type: Literal
    """Dataset type, must be 'ko' """
    car_type: Tuple[Literal['Car', 'Truck', 'Van'], ...] = ("Car")
    """Car type to process, 'Car', 'Truck' and 'Van' are valid"""


    def __post_init__(self)->None:
        assert self.dataset_type == 'ko'
        self.seqs = ['00']
        
        self.car_instances = list()
        
        for seq in self.seqs:

            frame_dict, oid_dict = read_kitti_obj_label(
                Path.joinpath(self.dataset_dir, 'training/label_2'), self.car_type)
            self.car_instances += generate_file_path_list_from_obj_dict(frame_dict, oid_dict, seq)
        
        sam_checkpoint = "dependencies/segment-anything/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)
        self.oid = 0

    def __len__(self)->int:
        return len(self.car_instances)

    def __getitem__(self, idx)->Tuple[List, List, List]:
        frame = self.car_instances[idx]
        sid = frame['sid']
        fid = frame['fid']
        image_path = Path.joinpath(self.dataset_dir,
                                   f'training/image_2/{fid:06d}.png')
        image = imageio.v2.imread(str(image_path))
        self.predictor.set_image(image)

        input_boxes = list()
        idx_helper = list()

        for idx, car_obj in enumerate(frame['car_objs']):
            input_box = car_obj.box2d
            xmin, ymin, xmax, ymax = input_box
            # filter out small patches
            if (ymax - ymin) < 64 or (xmax - xmin) < 32 or car_obj.t[2] > 50.0:
                pass
            else:
                input_boxes.append(input_box)
                idx_helper.append(idx)

        patches = list()
        masked_patches = list()
        json_segment = list()
        if len(input_boxes) == 0:
            return (patches, masked_patches, json_segment)
        else:
            input_boxes = np.array(input_boxes)

            transformed_boxes = self.predictor.transform.apply_boxes(input_boxes, image.shape[:2])
            transformed_boxes = torch.Tensor(transformed_boxes).to(self.predictor.device)

            masks, ious, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            for idx, iou in enumerate(ious):
                # filter out the patches that is heavily occluded and low confidence
                this_obj = frame['car_objs'][idx_helper[idx]]
                obj_xmin, obj_ymin, obj_xmax, obj_ymax = np.array(this_obj.box2d, dtype=int)
                intersec = masks[idx].sum() / ((obj_xmax+1 - obj_xmin) * (obj_ymax + 1 - obj_ymin))
                if float(iou) > 0.6 and intersec.item() > 0.6:
                    fid = frame['fid']
                    sid = frame['sid']
                    oid = self.oid
                    self.oid += 1
                    mask_to_save = masks[idx, :, obj_ymin: obj_ymax+1, 
                                         obj_xmin: obj_xmax+1].permute((1,2,0)).cpu().numpy()
                    patch_to_save = image[obj_ymin: obj_ymax+1, obj_xmin: obj_xmax+1, :]
                    patches.append(patch_to_save)
                    masked_patches.append(mask_to_save)
                    info = self.generate_kitti_obj_dict(this_obj, sid, fid, oid)
                    json_segment.append(info)
        
        return patches, masked_patches, json_segment
    
    def generate_kitti_obj_dict(self, obj: Object3d, sid: int, fid: int, oid: int):
        '''generate the vkitti info json
        '''
        proj_mat = Calibration(str(Path.joinpath(self.dataset_dir, f'training/calib/{fid:06}.txt')))
        file_name = f'{self.dataset_type}_{sid:02d}{fid:05d}{oid:05d}'
        img_file_name = f'../kitti-obj/training/image_2/{fid:06d}.png'
        patch_file_name = file_name + '_patch.png'
        mask_file_name = file_name + '_mask.png'
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
                          }
        return one_patch


def generate_file_path_list_from_obj_dict(frame_dict, oid_dict, sid) -> List:
    '''parse_
    '''
    result = list()
    for frame_id, car_obj in frame_dict.items():
        result.append({'fid':frame_id,
                       'car_objs':car_obj,
                       'oid':oid_dict[frame_id],
                       'sid':int(sid[-2:])})
    return result

def read_vkitti_tracking_label(path:Path, car_type) -> Tuple[Dict, Dict]:
    '''read labels from vkitti dataset
    '''
    frame_dict={}
    oid_dict={}
    camera_id = [0]
    bbox_path = Path.joinpath(path, 'bbox.txt')
    pose_path = Path.joinpath(path, 'pose.txt')
    info_path = Path.joinpath(path, 'info.txt')

    id_to_type_map = parse_vkitti_type_name(info_path)
 
    with open(bbox_path, encoding='UTF-8') as f1, \
    open(pose_path, encoding='UTF-8') as f2:
        
        bbox_file = f1.readlines()
        pose_file = f2.readlines()
        bbox_file = bbox_file[1:]
        pose_file = pose_file[1:]

        for idx, bbox_line in enumerate(bbox_file):
            pose_line = pose_file[idx]
            bbox_line, pose_line = bbox_line.split(), pose_line.split()
            this_cam_id = int(bbox_line[1])
            this_obj_id = int(bbox_line[2])
            this_type = id_to_type_map[this_obj_id]
            if this_cam_id in camera_id and this_type in car_type:
                xmin, xmax, ymin, ymax = bbox_line[3:7]
                trunc, occ = bbox_line[8:10]
                alpha, width, height, length = pose_line[3:7]
                frame_id = int(pose_line[0])
                ob_id = int(pose_line[2])
                t_x, t_y, t_z, yaw = pose_line[-6:-2]
                line = ' '.join([this_type, trunc, occ,
                                 alpha, xmin, ymin, xmax, ymax,
                                 height, width, length,
                                 t_x, t_y, t_z, yaw])

                if frame_id in frame_dict.keys():
                    frame_dict[frame_id].append(Object3d(line))
                    oid_dict[frame_id].append(ob_id)
                else:
                    frame_dict[frame_id] = [Object3d(line)]
                    oid_dict[frame_id] = [ob_id]
    f1.close()
    f2.close()
    return frame_dict, oid_dict

def parse_vkitti_type_name(path)->Dict:
    '''Parse the car type from info.txt file
    '''
    result = {}
    with open(path, encoding='UTF-8') as f1:
        lines = f1.readlines()
        lines = lines[1:]
        for line in lines:
            line = line.split()
            car_id, car_type = int(line[0]), line[1]
            result[car_id] = car_type
    return result

def read_kitti_mot_label(path, car_type):
    '''Read labels from kitti mot sequences
    '''

    frame_dict={}
    oid_dict={}

    with open(path, encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.split()
            this_name = line[2]
            frame_id = int(line[0])
            ob_id = int(line[1])

            if this_name in car_type:
                line = line[2:]

                if frame_id in frame_dict.keys():
                    frame_dict[frame_id].append(Object3d(' '.join(line)))
                    oid_dict[frame_id].append(ob_id)
                else:
                    frame_dict[frame_id] = [Object3d(' '.join(line))]
                    oid_dict[frame_id] = [ob_id]

    return frame_dict, oid_dict       




class VkittiCalibration():
    ''' Return the projection matrix for vkitti
    '''
    def __init__(self):
        self.P = np.array([725.0087, 0., 620.5, np.finfo(float).eps,
                  0., 725.0087, 187., np.finfo(float).eps,
                  0., 0., 1., np.finfo(float).eps], dtype=float).reshape((3,4))
        self.c_u = self.P[0,2]
        self.c_v = self.P[1,2]
        self.f_u = self.P[0,0]
        self.f_v = self.P[1,1]
        self.b_x = self.P[0,3]/(-self.f_u) # relative
        self.b_y = self.P[1,3]/(-self.f_v)



        

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr






def read_kitti_obj_label(path, car_type):
    '''Read kitti object labels
    '''

    frame_dict={}
    oid_dict={}

    frames = [i for i in os.listdir(path) if 'txt' in i]
    ob_id = int(0)


    for frame_name in frames:

        with open(Path(path).joinpath(frame_name), encoding='UTF-8') as f:
            frame_id = int(frame_name[:-4])
            for line in f.readlines():
                line = line.split()
                this_name = line[0]

                if this_name in car_type:
                    if frame_id in frame_dict.keys():
                        frame_dict[frame_id].append(Object3d(' '.join(line)))
                        oid_dict[frame_id].append(ob_id)
                    else:
                        frame_dict[frame_id] = [Object3d(' '.join(line))]
                        oid_dict[frame_id] = [ob_id]
                    ob_id += 1

    return frame_dict, oid_dict

def collate_fn_dataloader(data):
    '''collate fn for the dataloader
    '''
    return data[0]
