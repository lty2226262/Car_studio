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

"""
Semantic dataset.
"""

from typing import Dict

import numpy as np
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.rich_utils import CONSOLE

from car_studio.data.utils.data_utils import get_image_tensor_from_path


class CarPatchDataset(InputDataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert "src_image_files" in dataparser_outputs.metadata.keys()
        assert "xyxy_masks" in dataparser_outputs.metadata.keys()
        self.src_image_files = self.metadata["src_image_files"]
        self.xyxy_masks = self.metadata["xyxy_masks"]
        self.object_ids = self.metadata["object_ids"]

    def get_metadata(self, data: Dict) -> Dict:
        # handle source images, mask and object_id
        src_img_filepath = self.src_image_files[data["image_idx"]]
        src_img = get_image_tensor_from_path(src_img_filepath)
        xyxy_mask = self.xyxy_masks[data["image_idx"]]
        object_id = self.object_ids[data["image_idx"]]
        return {"xyxy_mask": xyxy_mask, "src_img": src_img, "object_id": object_id}


class CarPatchDatasetMultiViewOnly(InputDataset):
    """Dataset that returns images in multi view format.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert "src_image_files" in dataparser_outputs.metadata.keys()
        assert "xyxy_masks" in dataparser_outputs.metadata.keys()
        self.src_image_files = self.metadata["src_image_files"]
        self.xyxy_masks = self.metadata["xyxy_masks"]
        self.object_ids = self.metadata["object_ids"]
        self.split = self.metadata["split"]
        obj_id_lookup_table = {}
        for i, obj_id in enumerate(self.object_ids):
            if obj_id not in obj_id_lookup_table.keys():
                obj_id_lookup_table[obj_id] = [i]
            else:
                obj_id_lookup_table[obj_id].append(i)
        
        CONSOLE.print(f'Instance counts (single view + multi view): {len(obj_id_lookup_table)}')
        if self.split == 'train':
            self._instance_lists = self._filter_out_signle_view_imgs(obj_id_lookup_table=obj_id_lookup_table)
        else:
            self._instance_lists = self._keep_all_view_imgs(obj_id_lookup_table=obj_id_lookup_table)
        assert len(self._instance_lists) > 0 or \
            len(obj_id_lookup_table) == 0, "no multi views in this dataset."

        # Lookup table here, filter out the and generate other view patch 

    def get_metadata(self, data: Dict) -> Dict:
        # handle source images, mask and object_id
        src_img_filepath = self.src_image_files[data["image_idx"]]
        src_img = get_image_tensor_from_path(src_img_filepath)
        xyxy_mask = self.xyxy_masks[data["image_idx"]]
        object_id = self.object_ids[data["image_idx"]]
        return {"xyxy_mask": xyxy_mask, "src_img": src_img, "object_id": object_id}
    
    def get_data(self, object_idx: int) -> Dict:
        avail_view_indices_lists = self._instance_lists[object_idx].values()
        avail_view_indices = next(iter(avail_view_indices_lists))
        replace = True if len(avail_view_indices) < 2 else False
        src_img_idx, corressponding_img_idx = np.random.choice(avail_view_indices, 2,
                                                       replace=replace)
        data = super().get_data(src_img_idx)
        corressponding_img = self.get_image(corressponding_img_idx)
        data['corresponding_image'] = corressponding_img
        return data
        
    def __len__(self):
        return len(self._instance_lists)

    def _filter_out_signle_view_imgs(self, obj_id_lookup_table: Dict) -> list:
        result = []
        for key, value in obj_id_lookup_table.items():
            if len(value) > 1:
                result.append({key : value})

        CONSOLE.print(f'Instance counts (multi view): {len(result)}')
        return result
    
    def _keep_all_view_imgs(self, obj_id_lookup_table: Dict) -> list:
        result = []
        for key, value in obj_id_lookup_table.items():
            result.append({key : value})
        CONSOLE.print(f'keep single+multi view: {len(result)}')
        return result

class CarPatchDatasetSingleMultiViews(CarPatchDatasetMultiViewOnly):
    """Dataset that returns images in multi view + single view format.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """
    def _filter_out_signle_view_imgs(self, obj_id_lookup_table: Dict) -> list:
        return super()._keep_all_view_imgs(obj_id_lookup_table)
