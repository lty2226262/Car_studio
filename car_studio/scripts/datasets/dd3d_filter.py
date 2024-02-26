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

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tyro
from nerfstudio.utils.io import load_from_json, write_to_json
from pyquaternion import Quaternion
from tqdm import tqdm

BOX3D_CORNER_MAPPING = np.array([
    [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
    [0, -1, -1, 0, 0, -1, -1, 0],
    [0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5]
])

@dataclass
class DD3DFilter:
    """count the obj numbers and sort."""
    dvm_data_dir: Path = Path('./data/dvm_cars/')
    """Path to dataset."""
    result_json: Path = Path('./outputs/filter_dd3d/20230615/115304/kitti_format_predictions.json')
    out_dir: Path = Path('./outputs/filter_dd3d/')

    def main(self) -> None:
        timestr = time.strftime("%Y%m%d/%H%M%S")
        out_json_filename = self.out_dir / f'{timestr}/filtered_result.json'
        df = pd.read_csv(str(self.dvm_data_dir / 'Image_table.csv'), header=0,
                             index_col=2)
        d = df.to_dict('index')
        result = load_from_json(self.result_json)
        diff_value = []
        new_diff_value = []
        filtered_result = []
        small_patch_cnt = 0
        K = np.array([[721.5377, 0., 150.0],
                      [0., 721.5377, 150.0],
                      [0., 0., 1.]])
        
        for instance in tqdm(result):
            filename = instance['file_name'].rsplit('/', maxsplit=1)[-1]
            yaw = instance['rot_y']
            if filename not in d.keys():
                print(f'skip {filename} cannot validate')
                continue
            view_point = d[filename][' Predicted_viewpoint']
            diff = self._get_diff(yaw=yaw, view_point=view_point)
            diff_value.append(diff)

            if diff < 20 and instance['score'] > 0.6 and instance['score_3d'] > 0.6:
                quat=Quaternion(instance['q_wxyz'])
                T_bev_kitti = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
                q_bev_kitti = Quaternion(matrix=T_bev_kitti)
                new_quat = quat * q_bev_kitti
                new_rot = new_quat.rotation_matrix
                new_yaw = np.arctan2(-new_rot[2,0], new_rot[2,2])

                new_diff = self._get_diff(yaw=new_yaw, view_point=view_point)

                new_diff_value.append(new_diff)
                if new_diff < 20.:
                    new_instance = {
                        'patch': '',
                        'mask': '',
                        'image_file': '../' + str(Path(instance['file_name']).relative_to('/home/joey/code/car-studio/data')),
                        'fl_x': 721.5377,
                        'fl_y': 721.5377,
                        'cx': 150.0,
                        'cy': 150.0,
                        'cam_tx': 0.0,
                        'cam_ty': 0.0,
                        'cam_tz': 0.0,
                        'xmin': instance['l'],
                        'xmax': instance['r'] - 1,
                        'ymin': instance['t'],
                        'ymax': instance['b'] - 1,
                        'height': instance['H'],
                        'width': instance['W'],
                        'length': instance['L'],
                        'obj_x': instance['x'],
                        'obj_y': instance['y'],
                        'obj_z': instance['z'],
                        'yaw': instance['rot_y'],
                        'w': 300,
                        'h': 300,
                    }
                    if new_instance['ymax'] - new_instance['ymin'] < 64 or new_instance['xmax'] - new_instance['xmin'] < 64 \
                        or new_instance['ymax'] < 150 or new_instance['ymin'] > 150 or new_instance['xmin'] > 100 or \
                        new_instance['xmax'] < 200:
                        file_path = str(Path(instance['file_name']))
                        pos = np.array([instance['x'], instance['y'], instance['z']]) 
                        c = np.cos(yaw)
                        s = np.sin(yaw)
                        ori_rot = np.array([[c, 0, s],[0, 1, 0], [-s, 0, c]])
                        lhw = np.array([instance['L'], instance['H'], instance['W']])[:, None]
                        corners = lhw * BOX3D_CORNER_MAPPING
                        homo = np.pad(corners, ((0,1), (0, 0)), mode='constant', constant_values=1.0)
                        Trans_kitti = np.identity(4).copy()
                        Trans_kitti[:3, :3] = ori_rot
                        Trans_kitti[:3, 3] = pos.copy()
                        d_corners_kitti = np.dot(Trans_kitti, homo)[:3, :]
                        pixels_kitti = np.dot(K, d_corners_kitti)
                        pix_x_kitti = pixels_kitti[0, :] / pixels_kitti[2, :]
                        pix_y_kitti = pixels_kitti[1, :] / pixels_kitti[2, :]
                        tmp_img = cv2.imread(file_path)
                        for i in range(4):
                            cv2.line(
                                tmp_img, (int(pix_x_kitti[i]), int(pix_y_kitti[i])), (int(pix_x_kitti[i + 1]), int(pix_y_kitti[i + 1])),
                                (0, 255 ,0),
                                thickness=5
                            )
                            cv2.line(
                                tmp_img, (int(pix_x_kitti[i]), int(pix_y_kitti[i])), (int(pix_x_kitti[i + 4]), int(pix_y_kitti[i + 4])),
                                (0, 0 ,255),
                                thickness=5
                            )
                        cv2.imwrite('./test.jpg', tmp_img)
                        # break here to watch the outliers.
                        small_patch_cnt +=1
                    else:
                        filtered_result.append(new_instance)
                #     file_path = str(Path(instance['file_name']))
                #     print(str(Path(instance['file_name']).relative_to('/home/joey/code/car-studio/data')))
                #     print("view point:", view_point)
                #     print("yaw:", instance['rot_y'])
                #     print("new yaw:", new_yaw)
                #     c = np.cos(yaw)
                #     s = np.sin(yaw)
                #     ori_rot = np.array([[c, 0, s],[0, 1, 0], [-s, 0, c]])
                #     ori_yaw = np.arctan2(-ori_rot[2,0], ori_rot[2, 2])
                #     print('ori yaw:', ori_yaw)
                #     print("ori rot:", ori_rot)
                #     print("new rot:", new_rot)
                #     print('hhhhhh')

                #     lhw = np.array([instance['L'], instance['H'], instance['W']])[:, None]
                #     corners = lhw * BOX3D_CORNER_MAPPING
                #     homo = np.pad(corners, ((0,1), (0, 0)), mode='constant', constant_values=1.0)
                #     pos = np.array([instance['x'], instance['y'], instance['z']])
                #     Trans = np.identity(4)
                #     Trans[:3, :3] = quat.rotation_matrix.copy()
                #     Trans[:3, 3] = pos.copy()
                #     d_corners = np.dot(Trans, homo)[:3, :]

                #     Trans_kitti = np.identity(4).copy()
                #     Trans_kitti[:3, :3] = ori_rot
                #     Trans_kitti[:3, 3] = pos.copy()
                #     d_corners_kitti = np.dot(Trans_kitti, homo)[:3, :]

                #     K = np.array([[721.5377, 0., 150.0],
                #                   [0., 721.5377, 150.0],
                #                   [0., 0., 1.]])
                    
                #     pixels = np.dot(K, d_corners)
                #     pixels_kitti = np.dot(K, d_corners_kitti)
                #     pix_x = pixels[0, : ] / pixels[2, :]
                #     pix_y = pixels[1, : ] / pixels[2, :]
                #     pix_x_kitti = pixels_kitti[0, :] / pixels_kitti[2, :]
                #     pix_y_kitti = pixels_kitti[1, :] / pixels_kitti[2, :]
                #     tmp_img = cv2.imread(file_path)
                #     for i in range(4):
                #         cv2.line(
                #             tmp_img, (int(pix_x[i]), int(pix_y[i])), (int(pix_x[i + 4]), int(pix_y[i + 4])),
                #             (255, 0 ,0),
                #             thickness=5
                #         )
                #         cv2.line(
                #             tmp_img, (int(pix_x[i]), int(pix_y[i])), (int(pix_x[i + 1]), int(pix_y[i + 1])),
                #             (0, 255 ,0),
                #             thickness=5
                #         )
                #         cv2.line(
                #             tmp_img, (int(pix_x_kitti[i]), int(pix_y_kitti[i])), (int(pix_x_kitti[i + 4]), int(pix_y_kitti[i + 4])),
                #             (0, 0 ,255),
                #             thickness=5
                #         )
                #     cv2.imwrite('./test.jpg', tmp_img)
                #     print('hhhhh')


            # if diff < 20 and instance['score'] > 0.6 and instance['score_3d'] > 0.6:
            #     # filtered_result.append(instance)
            #     quat=Quaternion(instance['q_wxyz'])
            #     T_bev_kitti = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
            #     q_bev_kitti = Quaternion(matrix=T_bev_kitti)
            #     new_quat = quat * q_bev_kitti
            #     new_rot = new_quat.rotation_matrix
            #     new_yaw = np.arctan2(new_rot[2,2], -new_rot[2,0])

            #     yaw = instance['rot_y']
            #     c = np.sin(yaw)
            #     s = np.cos(yaw)
            #     ori_rot = np.array([[c, 0, s],[0, 1, 0], [-s, 0, c]])
            #     ori_yaw = np.arctan2(ori_rot[2,2], -ori_rot[2,0])


        print('small_path_cnt:',small_patch_cnt)
        out_json_filename.parent.mkdir(parents=True, exist_ok=True)
        write_to_json(out_json_filename, {'dataset_name': 'dvm_car_studio', 'instances': filtered_result})
        diff_value = np.array(diff_value)
        hist, edges = np.histogram(
            diff_value,
            bins=36,
            range=(0, 360),
            density=False
        )
        print('yaw_diff:', hist, edges)

        new_diff_value = np.array(new_diff_value)
        hist, edges = np.histogram(
            new_diff_value,
            bins=36,
            range=(0, 360),
            density=False
        )
        print('new_yaw_diff:', hist, edges)

        
        
    def _get_diff(self, yaw:float, view_point:float) -> float:
        inv = 360 - view_point + 90
        if inv > 180:
            normalize = inv - 360
        else:
            normalize = inv
        
        radian = float(normalize) / 180. * np.pi
        diff = np.abs(yaw - radian) / np.pi * 180.
        if diff >= 180:
            diff = np.abs(diff - 360)
        return diff

        # print(d)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(DD3DFilter).main()


if __name__ == "__main__":
    entrypoint()
