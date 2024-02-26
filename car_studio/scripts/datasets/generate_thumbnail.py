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

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import imageio
import numpy as np
import torch
import tyro
from nerfstudio.utils.io import load_from_json, write_to_json
from PIL import Image, ImageDraw
from tqdm import tqdm

from car_studio.utils.kitti_utils import (ProcessKittiMot, ProcessKittiObj,
                                          ProcessVKitti, collate_fn_dataloader)

BOX3D_CORNER_MAPPING = np.array([
    [1, 1, 1, 1, -1, -1, -1, -1],
    [0, -1, -1, 0, 0, -1, -1, 0],
    [1, 1, -1, -1, 1, 1, -1, -1]
])

@dataclass
class GenerateThumbnail:
    """Use cuboid detections to render masks for dynamic objects."""

    data_dir: Path = Path('./data/car_studio/patch/')
    """Path to dataset."""
    seed: int = 114514
    mode: Literal['Mask', 'Image', 'Detection'] = 'Detection'

    def main(self) -> None:
        np.random.seed(self.seed)
        keywords = ['ko', 'km', 'dv_0*6*4']
        json_file = load_from_json(self.data_dir / '../ko.json')['instances'] + \
            load_from_json(self.data_dir / '../km.json')['instances'] + \
            load_from_json(self.data_dir / '../dv.json')['instances']
        print('finish json loaded.')
        infos = []
        for keyword in keywords:
            idx = 0
            for file in glob.glob(str(self.data_dir) + '/' + keyword+ '*'):
                if idx > 1000:
                    break
                if self.mode == 'Mask':
                    file = file.replace('patch', 'mask')
                tmp_img = Image.open(file)
                ratio = float(tmp_img.width) / tmp_img.height
                if ratio > 2.0 or ratio < 1.5:
                    continue
                info = {
                    'f': tmp_img,
                    'w': tmp_img.width,
                    'h': tmp_img.height,
                    'r': ratio,
                    'p': file.split('/', maxsplit=2)[-1],
                }
                infos.append(info)
                idx += 1
        all_list_idx = np.random.choice(len(infos), 289,replace=False)
        if self.mode == 'Mask':
            canvas = np.zeros((90 * 17, 120 * 17), dtype=bool)
        else:
            canvas = np.zeros((90 * 17, 120 * 17, 3), dtype=np.uint8)
        for idx, i in enumerate(all_list_idx):
            if self.mode == 'Detection':
                patch_file = infos[i]['p']
                result = None
                for instance in json_file:
                    if instance['patch'] == patch_file:
                        result = instance
                        break
                assert result is not None
                tx = result['obj_x']
                ty = result['obj_y']
                tz = result['obj_z']
                l = result['length']
                h = result['height']
                w = result['width']
                ry = result['yaw']
                imshape_h = result['h']
                imshape_w = result['w']
                P = np.array([[result['fl_x'], 0, result['cx'], result['cam_tx']],
                    [0, result['fl_y'], result['cy'], result['cam_ty']],
                    [0, 0, 1, result['cam_tz']]])
                corners = self._get_corners([tx, ty, tz, l, h, w, ry])
                pixels = self._project_to_image(corners, P)
                pixel_offset = np.array([result['xmin'], result['ymin']])[None,:]
                pixels = pixels - pixel_offset
                img = np.array(infos[i]['f'])
                self._pretty_render_3d_box(pixels, img, line_thickness=3, color=(0, 255, 0))
                pil_img = Image.fromarray(img)
                infos[i]['f'] = pil_img

            img_x = idx % 17
            img_y = idx // 17
            canvas[img_y * 90: (img_y + 1) * 90, img_x * 120 : (img_x + 1) * 120, ...] = np.array(infos[i]['f'].resize((120, 90)))
        image = Image.fromarray(canvas)
        image.save(f'./thumbtail_{self.mode}.png')

        print(infos)

    def _get_corners(self, obj):
        if isinstance(obj, list):
            tx, ty, tz, l, h, w, ry = obj
        else:
            tx, ty, tz, l, h, w, ry = list(obj)
        
        # 3d bounding box corners

        x_corners = BOX3D_CORNER_MAPPING[0,:] * l / 2
        y_corners = BOX3D_CORNER_MAPPING[1,:] * h
        z_corners = BOX3D_CORNER_MAPPING[2,:] * w / 2
    
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

    
    def _pretty_render_3d_box(
        self,
        corners,
        image,
        # camera
        line_thickness=3,
        color=None,
    ):
        """Render the bounding box on the image. NOTE: CV2 renders in place.

        Parameters
        ----------
        box3d: GenericBoxes3D

        image: np.uint8 array
            Image (H, W, C) to render the bounding box onto. We assume the input image is in *RGB* format

        K: np.ndarray
            Camera used to render the bounding box.

        line_thickness: int, default: 1
            Thickness of bounding box lines.

        font_scale: float, default: 0.5
            Font scale used in text labels.

        draw_axes: bool, default: False
            Whether or not to draw axes at centroid of box.
            Note: Bounding box pose is oriented such that x-forward, y-left, z-up.
            This corresponds to L (length) along x, W (width) along y, and H
            (height) along z.

        draw_text: bool, default: False
            If True, renders class name on box.
        """

        # Draw the sides (first)
        for i in range(4):
            cv2.line(
                image, (int(corners[i][0]), int(corners[i][1])), (int(corners[i + 4][0]), int(corners[i + 4][1])),
                color,
                thickness=line_thickness
            )
        # Draw front (in red) and back (blue) face.
        cv2.polylines(image, [corners[:4].astype(np.int32)], True, color, thickness=line_thickness)
        cv2.polylines(image, [corners[4:].astype(np.int32)], True, color, thickness=line_thickness)

        front_face_as_polygon = corners[:4].ravel().astype(int).tolist()
        self._fill_color_polygon(image, front_face_as_polygon, color, alpha=0.5)
        return image
    
    def _fill_color_polygon(self, image, polygon, color, alpha=0.5):
        """Color interior of polygon with alpha-blending. This function modified input in place.
        """
        _mask = Image.new('L', (image.shape[1], image.shape[0]), 0)
        ImageDraw.Draw(_mask).polygon(polygon, outline=1, fill=1)
        mask = np.array(_mask, dtype=bool)
        for c in range(3):
            channel = image[:, :, c]
            channel[mask] = channel[mask] * (1. - alpha) + color[c] * alpha

                    



def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(GenerateThumbnail).main()


if __name__ == "__main__":
    entrypoint()
