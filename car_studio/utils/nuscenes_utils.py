# 2023 Tianyu LIU [tliubk@connect.ust.hk]. All rights reserved.
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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import imageio
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


@dataclass
class ProcessNuscenes():
    ''' Parallel processing v-kitti dataset
    '''
    dataset_dir: Path
    """Path to dataset root"""
    dataset_type: Literal
    """Dataset type, must be 'ns' """
    dict_list: Dict
    """Dataset file lists"""

    def __post_init__(self)->None:
        assert self.dataset_type == "ns"
        sam_checkpoint = "dependencies/segment-anything/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def __len__(self)->int:
        return len(self.dict_list)

    def __getitem__(self, idx)->Tuple[List, List, List]:
        info = self.dict_list[idx]
        patches = list()
        masked_patches = list()
        json_segment = list()
        input_boxes = list()

        xmin = info['xmin']
        xmax = info['xmax']
        ymin = info['ymin']
        ymax = info['ymax']

        if (xmax - xmin) < 32 or (ymax - ymin) < 64 or info['obj_z'] > 50.0:
            pass
        else:
            image_path = Path.joinpath(self.dataset_dir, info['image_file'])
            image = imageio.v2.imread(str(image_path))
            self.predictor.set_image(image)
            input_boxes.append(np.array([xmin, ymin, xmax, ymax]))

        if len(input_boxes) > 0:
            input_boxes = np.array(input_boxes)

            transformed_boxes = self.predictor.transform.apply_boxes(input_boxes, image.shape[:2])
            transformed_boxes = torch.Tensor(transformed_boxes).to(self.predictor.device)

            masks, ious, _ = self.predictor.predict_torch(
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
                patches.append(patch_to_save)
                masked_patches.append(mask_to_save)
                json_segment.append(info)
            else:
                pass
        else:
            pass

        return patches, masked_patches, json_segment
