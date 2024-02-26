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

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import tinycudann as tcnn
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.field_components.encodings import (NeRFEncoding,
                                                   TensorVMEncoding,
                                                   TriplaneEncoding)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler
from nerfstudio.model_components.renderers import (AccumulationRenderer,
                                                   DepthRenderer, RGBRenderer)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.vanilla_nerf import VanillaModelConfig
from nerfstudio.utils import colormaps, colors, misc
from torch import Tensor
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from car_studio.field_components.encodings import TriplaneZSymmEncoding
from car_studio.field_components.resnet_image_encoder import ResnetImageEncoder
from car_studio.fields.carnerfsymm_field import CarNeRFSymmField
from car_studio.model_components.custom_ray_samplers import (
    CustomPDFSampler, CustomUniformSampler)


@dataclass 
class CarNerfSymmModelConfig(VanillaModelConfig):
    """Car Nerf Config"""

    _target: Type = field(default_factory=lambda: CarNerfSymmModel)
    init_resolution: int = 128
    """initial render resolution"""
    final_resolution: int = 300
    """final render resolution"""
    upsampling_iters: Tuple[int, ...] = (2000, 3000, 4000, 5500, 7000)
    """specifies a list of iteration step numbers to perform upsampling"""
    num_den_components: int = 16
    """Number of components in density encoding"""
    num_color_components: int = 48
    """Number of components in color encoding"""
    num_samples: int = 50
    """Number of samples in field evaluation"""
    num_uniform_samples: int = 200
    """Number of samples in density evaluation"""
    background_color: Literal["random", "black", "white"] = "white"
    enable_collider: bool = False
    appearance_dim: int = 27

class CarNerfSymmModel(Model):
    """Car Nerf model

    Args:
        config: car nerf configuration to instantiate model
    """

    config: CarNerfSymmModelConfig

    def __init__(self, config: ModelConfig, scene_box: SceneBox, num_train_data: int, **kwargs) -> None:
        super().__init__(config, scene_box, num_train_data, **kwargs)

    def update_to_step(self, step: int) -> None:
        if step < self.upsampling_iters[0]:
            return

        new_iters = list(self.upsampling_iters) + [step + 1]
        new_iters.sort()

        index = new_iters.index(step + 1)
        new_grid_resolution = self.upsampling_steps[index - 1]

        self.field.density_encoding.upsample_grid(new_grid_resolution)  # type: ignore
        self.field.color_encoding.upsample_grid(new_grid_resolution)  # type: ignore


    def populate_modules(self):
        """Set the fields and modules."""
        
        # setting up encoder
        self.encoder = ResnetImageEncoder()
        self.init_resolution = self.config.init_resolution
        self.num_den_components = self.config.num_den_components
        self.num_color_components = self.config.num_color_components
        self.apperance_dim = self.config.appearance_dim
        self.upsampling_iters = self.config.upsampling_iters
        self.upsampling_steps = (
            np.round(
                np.exp(
                    np.linspace(
                        np.log(self.config.init_resolution),
                        np.log(self.config.final_resolution),
                        len(self.config.upsampling_iters) + 1,
                    )
                )
            )
            .astype("int")
            .tolist()[1:]
        )

        # setting up fields
        # position_encoding = NeRFEncoding(
        #     in_dim=self.apperance_dim,
        #     num_frequencies=2,
        #     min_freq_exp=0.0,
        #     max_freq_exp=2
        # )
        # position_encoding = tcnn.Encoding(
        #     n_input_dims=self.apperance_dim,
        #     encoding_config={"otype": "Frequency", "n_frequencies": 2},
        # )
        # direction_encoding = NeRFEncoding(
        #     in_dim=3, 
        #     num_frequencies=2,
        #     min_freq_exp=0,
        #     max_freq_exp=2)
        # direction_encoding = tcnn.Encoding(
        #     n_input_dims=3,
        #     encoding_config={"otype": "Frequency", "n_frequencies": 2},
        # )

        density_encoding = TriplaneEncoding(
            resolution=self.init_resolution,
            num_components=self.num_den_components
        )
        color_encoding = TriplaneEncoding(
            resolution=self.init_resolution,
            num_components=self.num_color_components
        )
        feature_encoding = NeRFEncoding(in_dim=self.apperance_dim,
                                        num_frequencies=2,
                                        min_freq_exp=0,
                                        max_freq_exp=2)
        direction_encoding = NeRFEncoding(in_dim=3,
                                          num_frequencies=2,
                                          min_freq_exp=0,
                                          max_freq_exp=2)
        

        self.field = CarNeRFSymmField(
            aabb=self.scene_box.aabb,
            feature_encoding=feature_encoding,
            direction_encoding=direction_encoding,
            density_encoding=density_encoding,
            color_encoding=color_encoding,
            appearance_dim=self.apperance_dim,
            use_integrated_encoding=True,
        )

        # samplers
        self.sampler_uniform = CustomUniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = CustomPDFSampler(num_samples=self.config.num_importance_samples,
                                      include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        if self.config.enable_collider:
            assert self.config.collider_params is not None
            self.collider = AABBBoxCollider(self.scene_box)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["encoder"] = list(self.encoder.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups
    
    def get_latents(self, img_patch: Tensor) -> Tensor:
        """ get the latents from encoder"""
        return self.encoder(img_patch)

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Tensor]:
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")
        
        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # First pass:
        dens = self.field.get_density(ray_samples_uniform)
        weights = ray_samples_uniform.get_weights(dens)
        coarse_intersect = ray_samples_uniform.frustums.starts[:,
                                                               0,0] < 1e9
        coarse_accumulation = self.renderer_accumulation(weights)

        # acc_mask = torch.where(coarse_accumulation < 0.0001, False, True).reshape(-1)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights)
        
        # fine field:
        field_outputs_fine = self.field.forward(
            ray_samples_pdf
        )

        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        accumulation = self.renderer_accumulation(weights_fine)
        rgb = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        rgb = torch.where(accumulation < 0, colors.WHITE.to(rgb.device), rgb)
        depth = self.renderer_depth(weights_fine, ray_samples_pdf)
        fine_intersect = ray_samples_pdf.frustums.starts[:,0,0] < 1e9


        # field_outputs_coarse = self.field.forward(ray_samples_uniform)
        # coarse_intersect = ray_samples_uniform.frustums.starts[:,
        #                                                        0,0] < 1e9
        # weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        # rgb_coarse = self.renderer_rgb(
        #     rgb=field_outputs_coarse[FieldHeadNames.RGB],
        #     weights=weights_coarse,
        # )
        # accumulation_coarse = self.renderer_accumulation(weights_coarse)
        # depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # # pdf sampling
        # ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        # fine_intersect = ray_samples_pdf.frustums.starts[:,0,0] < 1e9

        # # Second pass:
        # field_outputs_fine = self.field.forward(ray_samples_pdf)
        # weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        # rgb_fine = self.renderer_rgb(
        #     rgb=field_outputs_fine[FieldHeadNames.RGB],
        #     weights=weights_fine,
        # )
        # accumulation_fine = self.renderer_accumulation(weights_fine)
        # depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_fine": rgb,
            "accumulation_coarse": coarse_accumulation,
            "accumulation_fine": accumulation,
            "depth_fine": depth,
            "coarse_intersect": coarse_intersect,
            "fine_intersect": fine_intersect,
        }
        if torch.isnan(rgb[fine_intersect, ...]).sum():
            print('debug here')
        return outputs

        

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metric_dict = {}
        return metric_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, Tensor]:
        image_gt = batch["image"].to(self.device)
        mask_gt = batch["patch_mask"]
        
        # coarse_intersect = outputs['coarse_intersect']
        fine_intersect = outputs['fine_intersect']
        
        # coarse_rgb = outputs["rgb_coarse"][coarse_intersect, ...]
        # coarse_image_gt = image_gt[coarse_intersect, ...]
        # coarse_mask_gt = mask_gt[coarse_intersect, ...].to(torch.float)

        fine_rgb = outputs["rgb_fine"][fine_intersect, ...]
        fine_image_gt = image_gt[fine_intersect, ...]
        fine_mask_gt = mask_gt[fine_intersect, ...].to(torch.float)

        # rgb_loss_coarse = self.rgb_loss(coarse_rgb, coarse_image_gt * coarse_mask_gt)
        rgb_loss_fine = self.rgb_loss(fine_rgb, fine_image_gt * fine_mask_gt)
        if torch.isnan(rgb_loss_fine).sum():
            print('debug here')
        
        # coarse_accum = outputs['accumulation_coarse'][coarse_intersect, ...]
        fine_accum = outputs['accumulation_fine'][fine_intersect, ...]

        l1_parameters = []
        for parameter in self.field.density_encoding.parameters():
            l1_parameters.append(parameter.view(-1))
        l1_loss_reg = torch.abs(torch.cat(l1_parameters)).mean()

        # accum_loss_coarse = self.rgb_loss(coarse_accum, coarse_mask_gt)
        accum_loss_fine = self.rgb_loss(fine_accum, fine_mask_gt)
        loss_dict = {
                     "rgb_loss_fine": rgb_loss_fine,
                    #  "mask_loss_coarse": accum_loss_coarse,
                     "mask_loss_fine": accum_loss_fine,
                     "l1_reg": l1_loss_reg,
                     }

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        
        image = (batch["src_img"]*batch["patch_mask"]).to(outputs["rgb_fine"].device)
        xmin, ymin, xmax, ymax = batch["xyxy_msk"].squeeze(0)
        image = image[:,ymin:ymax+1, xmin:xmax+1,:].squeeze(0)
        # rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        rgb_fine = torch.nan_to_num(rgb_fine, 0)
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
        origin_mask = batch['patch_mask'][:,ymin:ymax+1, xmin:xmax+1,0].to(acc_coarse.device).to(torch.float32).squeeze(0).unsqueeze(-1)
        acc_mask = colormaps.apply_colormap(origin_mask)


        assert self.config.collider_params is not None
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_mask, acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
        rgb_fine = torch.clip(rgb_fine, min=0, max=1)

        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)

        assert isinstance(fine_ssim, torch.Tensor)
        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "fine_psnr": float(fine_psnr.item()),
            "fine_ssim": float(fine_ssim.item()),
            "fine_lpips": float(fine_lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict
    
    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        
        xmin, ymin, xmax, ymax = camera_ray_bundle.metadata['xyxy_mask'].squeeze(0).to(torch.long)
        image_height = ymax - ymin + 1
        image_width = xmax - xmin + 1
        num_rays = image_height * image_width

        outputs_lists = defaultdict(list)
        latents = self.get_latents(camera_ray_bundle.metadata['patch'].permute(0,3,1,2).to(self.device))
        camera_ray_bundle.metadata.pop('patch')
        camera_ray_bundle.metadata.pop('xyxy_mask')
        camera_ray_bundle.metadata.update({'latents': latents})
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle_and_latents(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        assert 'latents' in camera_ray_bundle.metadata.keys(), "no latents with the ray bundles"
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        
        xmin, ymin, xmax, ymax = camera_ray_bundle.metadata['xyxy_mask'].squeeze(0).to(torch.long)
        image_height = ymax - ymin + 1
        image_width = xmax - xmin + 1
        num_rays = image_height * image_width

        outputs_lists = defaultdict(list)
        camera_ray_bundle.metadata.pop('xyxy_mask')
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs
    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        # the callback that we want to run every X iterations after the training iteration
        def reinitialize_optimizer(
            self, training_callback_attributes: TrainingCallbackAttributes, step: int  # pylint: disable=unused-argument
        ):
            index = self.upsampling_iters.index(step)
            resolution = self.upsampling_steps[index]

            # upsample the position and direction grids
            self.field.density_encoding.upsample_grid(resolution)
            # reinitialize the encodings optimizer
            optimizers_config = training_callback_attributes.optimizers.config
            enc = training_callback_attributes.pipeline.get_param_groups()["fields"]
            lr_init = optimizers_config["fields"]["optimizer"].lr

            training_callback_attributes.optimizers.optimizers["fields"] = optimizers_config["fields"][
                "optimizer"
            ].setup(params=enc)
            if optimizers_config["fields"]["scheduler"]:
                training_callback_attributes.optimizers.schedulers["fields"] = (
                    optimizers_config["fields"]["scheduler"]
                    .setup()
                    .get_scheduler(
                        optimizer=training_callback_attributes.optimizers.optimizers["fields"], lr_init=lr_init
                    )
                )

        callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=self.upsampling_iters,
                func=reinitialize_optimizer,
                args=[self, training_callback_attributes],
            )
        ]
        return callbacks

