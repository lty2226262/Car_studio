from __future__ import annotations

from pathlib import Path

from nerfstudio.configs.base_config import MachineConfig
from nerfstudio.engine.optimizers import RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from car_studio.data.datamanagers.car_patch_datamanager import (
    CarPatchDataManager, CarPatchDataManagerConfig)
from car_studio.data.dataparsers.carstudio_dataparser import \
    CarstudioDataParserConfig
from car_studio.data.dataparsers.kittimot_dataparser import \
    KittiMotDataParserConfig
from car_studio.data.datasets.car_patch_dataset import (
    CarPatchDataset, CarPatchDatasetMultiViewOnly)
from car_studio.models.autorf import AutoRFModel, AutoRFModelConfig
from car_studio.models.car_nerf import CarNerfModel, CarNerfModelConfig
from car_studio.models.car_nerf_symm import (CarNerfSymmModel,
                                             CarNerfSymmModelConfig)
from car_studio.models.code_nerf import CodeNerfConfig, CodeNerfModel
from car_studio.models.pixelnerf import PixelNerfConfig, PixelNerfModel
from car_studio.models.pixelnerf_mlp import (PixelNerfMLPConfig,
                                             PixelNerfMLPModel)
from car_studio.pipelines.car_nerf_stage_one import \
    CarNerfStageOnePipelineConfig
from car_studio.pipelines.car_nerf_stage_two import \
    CarNerfStageTwoPipelineConfig
from car_studio.pipelines.codenerf_stage_one import \
    CodeNerfStageOnePipelineConfig
from car_studio.pipelines.codenerf_stage_two import \
    CodeNerfStageTwoPipelineConfig

CarNeRFStageOne = MethodSpecification(
    config=TrainerConfig(
        experiment_name="car-nerf-km-sv-with-mask",
        use_grad_scaler=False,
        method_name="car-nerf",
        mixed_precision=False,
        max_num_iterations=50000,
        # load_checkpoint=Path('/home/joey/code/car-studio/outputs/stage-one-km-sv-with-mask/car-nerf/2023-06-18_162045/nerfstudio_models/step-000529000.ckpt'),
        load_dir=None,
        steps_per_eval_batch=500,
        steps_per_eval_image=2000,
        steps_per_eval_all_images=250000,
        machine=MachineConfig(
            seed=114514,
            num_gpus=1,
            num_machines=1,
        ),
        pipeline=CarNerfStageOnePipelineConfig(
            datamanager=CarPatchDataManagerConfig(
                _target=CarPatchDataManager[CarPatchDataset],
                dataparser=CarstudioDataParserConfig(
                    subset=("km",)
                ),
                train_num_rays_per_batch=1024,
                train_num_images_to_sample_from=500,
                train_num_times_to_repeat_images=500,
                eval_num_images_to_sample_from=488,
                eval_num_times_to_repeat_images=488,
                n_instance_per_batch=3,
            ),
            model=CarNerfModelConfig(
                _target=CarNerfModel,
                loss_coefficients={"rgb_loss_fine": 1.0,
                                   "rgb_loss_coarse":0.1,
                                   "mask_loss_coarse": 0.1,
                                   "mask_loss_fine": 1.0,
                                   },
                num_coarse_samples=128,
                num_importance_samples=128,
                background_color="black",
                enable_collider=True,
                collider_params={
                    "near_plane": 0.1,
                    "far_plane": 50.0,
                }

            ),
            # load_pretrain_model_path=Path('outputs/car-nerf-dv-ko-sv-with-mask/car-nerf/2023-06-22_172508/nerfstudio_models/step-000500000.ckpt'),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "encoder": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000,),
            }
        },
        vis="wandb",
    ),
    description="Car NeRF implementations",
)

CarNeRFSymmStageOne = MethodSpecification(
    config=TrainerConfig(
        experiment_name="car-nerf-symm-km",
        use_grad_scaler=False,
        method_name="car-nerf-symm-stage-one",
        mixed_precision=False,
        max_num_iterations=5000000,
        load_dir=None,
        steps_per_eval_batch=10000,
        steps_per_eval_image=100,
        steps_per_eval_all_images=250000,
        machine=MachineConfig(
            seed=114514,
            num_gpus=1,
            num_machines=1,
        ),
        pipeline=CarNerfStageOnePipelineConfig(
            datamanager=CarPatchDataManagerConfig(
                _target=CarPatchDataManager[CarPatchDataset],
                dataparser=CarstudioDataParserConfig(
                    subset=("km",)
                ),
                train_num_rays_per_batch=1024,
                train_num_images_to_sample_from=500,
                train_num_times_to_repeat_images=500,
                eval_num_images_to_sample_from=494,
                eval_num_times_to_repeat_images=494,
                n_instance_per_batch=1,
            ),
            model=CarNerfSymmModelConfig(
                _target=CarNerfSymmModel,
                loss_coefficients={"rgb_loss_fine": 1.0,
                                   "rgb_loss_coarse":0.1,
                                   "mask_loss_coarse": 0.1,
                                   "mask_loss_fine": 1.0,
                                   },
                num_coarse_samples=128,
                num_importance_samples=128,
                background_color="black",
                enable_collider=True,
                collider_params={
                    "near_plane": 0.1,
                    "far_plane": 50.0,
                }

            ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "encoder": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000,),
            }
        },
        vis="wandb",
    ),
    description="Car NeRF Symm implementations",
)


AutoRFStageOne = MethodSpecification(
    config=TrainerConfig(
        experiment_name="autorf-stage-one-mv-km",
        use_grad_scaler=False,
        method_name="autorf",
        mixed_precision=False,
        max_num_iterations=500001,
        load_dir=None,
        steps_per_eval_batch=500,
        steps_per_eval_image=2000,
        steps_per_eval_all_images=250000,
        machine=MachineConfig(
            seed=114514,
            num_gpus=1,
            num_machines=1,
        ),
        pipeline=CarNerfStageOnePipelineConfig(
            datamanager=CarPatchDataManagerConfig(
                _target=CarPatchDataManager[CarPatchDataset],
                dataparser=CarstudioDataParserConfig(
                    subset=("km",)
                ),
                train_num_rays_per_batch=1024,
                train_num_images_to_sample_from=200,
                train_num_times_to_repeat_images=200,
                eval_num_images_to_sample_from=200,
                eval_num_times_to_repeat_images=200,
                n_instance_per_batch=3,
            ),
            model=AutoRFModelConfig(
                _target=AutoRFModel,
                loss_coefficients={"rgb_loss": 1.0},
                num_coarse_samples=128,
                background_color="black",
                enable_collider=True,
                collider_params={
                    "near_plane": 0.1,
                    "far_plane": 50.0,
                }

            )
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "encoder": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000,),
            }
        },
        vis="wandb",
    ),
    description="AutoRF implementations",
)


PixelNerfStageOne = MethodSpecification(
    config=TrainerConfig(
        experiment_name="pixelnerf-stage-one-mv-km",
        use_grad_scaler=False,
        method_name="pixelnerf",
        mixed_precision=False,
        max_num_iterations=500001,
        load_dir=None,
        steps_per_eval_batch=500,
        steps_per_eval_image=2000,
        steps_per_eval_all_images=250000,
        machine=MachineConfig(
            seed=114514,
            num_gpus=1,
            num_machines=1,
        ),
        pipeline=CarNerfStageOnePipelineConfig(
            datamanager=CarPatchDataManagerConfig(
                _target=CarPatchDataManager[CarPatchDatasetMultiViewOnly],
                dataparser=CarstudioDataParserConfig(
                    subset=("km",)
                ),
                train_num_rays_per_batch=1024,
                train_num_images_to_sample_from=500,
                train_num_times_to_repeat_images=500,
                eval_num_images_to_sample_from=500,
                eval_num_times_to_repeat_images=500,
                n_instance_per_batch=3,
            ),
            model=PixelNerfConfig(
                _target=PixelNerfModel,
                loss_coefficients={"coarse_rgb_loss": 0.1,
                                   "fine_rgb_loss": 1.0},
                num_coarse_samples=64,
                num_importance_samples=32,
                background_color="black",
                enable_collider=True,
                collider_params={
                    "near_plane": 0.1,
                    "far_plane": 50.0,
                },
            )
        ),
        optimizers={
            "coarse_fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "fine_fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "encoder": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000,),
            }
        },
        vis="wandb",
    ),
    description="PixelNerf implementations",
)

CodeNerfStageOne = MethodSpecification(
    config=TrainerConfig(
        experiment_name="codenerf-stage-one-ko-sv",
        use_grad_scaler=False,
        method_name="codenerf",
        mixed_precision=False,
        max_num_iterations=500000,
        load_dir=None,
        steps_per_eval_batch=500,
        steps_per_eval_image=2000,
        steps_per_eval_all_images=250000,
        steps_per_save=1000,
        machine=MachineConfig(
            seed=114514,
            num_gpus=1,
            num_machines=1,
        ),
        pipeline=CodeNerfStageOnePipelineConfig(
            datamanager=CarPatchDataManagerConfig(
                _target=CarPatchDataManager[CarPatchDataset],
                dataparser=CarstudioDataParserConfig(
                    subset=("ko",)
                ),
                train_num_rays_per_batch=1024,
                train_num_images_to_sample_from=500,
                train_num_times_to_repeat_images=500,
                eval_num_images_to_sample_from=494,
                eval_num_times_to_repeat_images=494,
                n_instance_per_batch=3,
            ),
            model=CodeNerfConfig(
                _target=CodeNerfModel,
                loss_coefficients={"rgb_loss": 1.0,},
                num_coarse_samples=64,
                num_importance_samples=32,
                background_color="black",
                enable_collider=True,
                collider_params={
                    "near_plane": 0.1,
                    "far_plane": 50.0,
                },
            )
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "latent_vectors": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000,),
            }
        },
        vis="wandb",
    ),
    description="PixelNerf implementations",
)

PixelNerfMLPStageOne = MethodSpecification(
    config=TrainerConfig(
        experiment_name="pixelnerf-mlp-stage-one-sv-ko",
        use_grad_scaler=False,
        method_name="pixelnerf-mlp",
        mixed_precision=False,
        max_num_iterations=500001,
        load_dir=None,
        steps_per_eval_batch=500,
        steps_per_eval_image=2000,
        steps_per_eval_all_images=250000,
        machine=MachineConfig(
            seed=114514,
            num_gpus=1,
            num_machines=1,
        ),
        pipeline=CarNerfStageOnePipelineConfig(
            datamanager=CarPatchDataManagerConfig(
                _target=CarPatchDataManager[CarPatchDataset],
                dataparser=CarstudioDataParserConfig(
                    subset=("ko",)
                ),
                train_num_rays_per_batch=1024,
                train_num_images_to_sample_from=200,
                train_num_times_to_repeat_images=200,
                eval_num_images_to_sample_from=200,
                eval_num_times_to_repeat_images=200,
                n_instance_per_batch=3,
            ),
            model=PixelNerfMLPConfig(
                _target=PixelNerfMLPModel,
                loss_coefficients={"coarse_rgb_loss": 1.0,
                                   "fine_rgb_loss": 1.0},
                num_coarse_samples=64,
                num_importance_samples=32,
                background_color="black",
                enable_collider=True,
                collider_params={
                    "near_plane": 0.1,
                    "far_plane": 50.0,
                },
            )
        ),
        optimizers={
            "coarse_fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "fine_fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "encoder": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000,),
            }
        },
        vis="wandb",
    ),
    description="PixelNerf-MLP implementations",
)


CarNeRFStageTwo = MethodSpecification(
    config=TrainerConfig(
        experiment_name="car-nerf-dv-ko-sv-seq-1-optimization",
        use_grad_scaler=False,
        method_name="car-nerf-stage-two",
        mixed_precision=False,
        max_num_iterations=200000,
        load_dir=None,
        steps_per_eval_batch=500,
        steps_per_eval_image=2000,
        steps_per_eval_all_images=500,
        save_only_latest_checkpoint=False,
        steps_per_save=10000,
        machine=MachineConfig(
            seed=114514,
            num_gpus=1,
            num_machines=1,
        ),
        pipeline=CarNerfStageTwoPipelineConfig(
            datamanager=CarPatchDataManagerConfig(
                _target=CarPatchDataManager[CarPatchDataset],
                dataparser=KittiMotDataParserConfig(
                    train_split_fraction = 1.0,
                    sequence_num= 6,
                    start_id= 1,
                    end_id=269,
                ),
                train_num_rays_per_batch=1024,
                train_num_images_to_sample_from=102,
                train_num_times_to_repeat_images=102,
                eval_num_images_to_sample_from=103,
                eval_num_times_to_repeat_images=103,
                n_instance_per_batch=3,
            ),
            model=CarNerfModelConfig(
                _target=CarNerfModel,
                loss_coefficients={"rgb_loss_fine": 1.0,
                                   "rgb_loss_coarse":0.1,
                                   "mask_loss_coarse": 0.1,
                                   "mask_loss_fine": 1.0,
                                   },
                num_coarse_samples=128,
                num_importance_samples=128,
                background_color="black",
                enable_collider=True,
                collider_params={
                    "near_plane": 0.1,
                    "far_plane": 50.0,
                }

            ),
            load_pretrain_model_path=Path('outputs/stage-one-ko-sv-with-mask/car-nerf/2023-06-16_133708/nerfstudio_models/step-000500000.ckpt'),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "latent_vectors": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000,),
            }
        },
        vis="wandb",
    ),
    description="Car NeRF stage two implementations",
)

PixelNerfMLPStageTwo = MethodSpecification(
    config=TrainerConfig(
        experiment_name="pixelnerf-mlp-stage-two-ko-train-km-seq-1-optimization",
        use_grad_scaler=False,
        method_name="pixelnerf-mlp-stage-two",
        mixed_precision=False,
        max_num_iterations=50000,
        load_dir=None,
        steps_per_eval_batch=1,
        steps_per_eval_image=2000,
        steps_per_eval_all_images=49999,
        save_only_latest_checkpoint=False,
        steps_per_save=1000,
        machine=MachineConfig(
            seed=114514,
            num_gpus=1,
            num_machines=1,
        ),
        pipeline=CarNerfStageTwoPipelineConfig(
            datamanager=CarPatchDataManagerConfig(
                _target=CarPatchDataManager[CarPatchDataset],
                dataparser=KittiMotDataParserConfig(
                    train_split_fraction = 1.0,
                    sequence_num= 1,
                    start_id= 73,
                    end_id=138,
                ),
                train_num_rays_per_batch=1024,
                train_num_images_to_sample_from=90,
                train_num_times_to_repeat_images=90,
                eval_num_images_to_sample_from=91,
                eval_num_times_to_repeat_images=91,
                n_instance_per_batch=3,
            ),
            model=PixelNerfMLPConfig(
                _target=PixelNerfMLPModel,
                loss_coefficients={"coarse_rgb_loss": 1.0,
                                   "fine_rgb_loss": 1.0},
                num_coarse_samples=64,
                num_importance_samples=32,
                background_color="black",
                enable_collider=True,
                collider_params={
                    "near_plane": 0.1,
                    "far_plane": 50.0,
                },
            ),
            load_pretrain_model_path=Path('outputs/pixelnerf-mlp-stage-one-sv-ko/pixelnerf-mlp/2023-06-26_083301/nerfstudio_models/step-000500000.ckpt')
        ),
        optimizers={
            "coarse_fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "fine_fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "latent_vectors": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000,),
            },
        },
        vis="wandb",
    ),
    description="PixelNerf-MLP stage two implementations",
)


AutoRFStageTwo = MethodSpecification(
    config=TrainerConfig(
        experiment_name="autorf-stage-two-sv-ko-seq-1-optimization",
        use_grad_scaler=False,
        method_name="autorf-stage-two",
        mixed_precision=False,
        max_num_iterations=50000,
        load_dir=None,
        steps_per_eval_batch=500,
        steps_per_eval_image=2000,
        steps_per_eval_all_images=49999,
        save_only_latest_checkpoint=False,
        steps_per_save=1000,
        machine=MachineConfig(
            seed=114514,
            num_gpus=1,
            num_machines=1,
        ),
        pipeline=CarNerfStageTwoPipelineConfig(
            datamanager=CarPatchDataManagerConfig(
                _target=CarPatchDataManager[CarPatchDataset],
                dataparser=KittiMotDataParserConfig(
                    train_split_fraction = 1.0,
                    sequence_num= 1,
                    start_id= 73,
                    end_id=138,
                ),
                train_num_rays_per_batch=1024,
                train_num_images_to_sample_from=90,
                train_num_times_to_repeat_images=90,
                eval_num_images_to_sample_from=91,
                eval_num_times_to_repeat_images=91,
                n_instance_per_batch=3,
            ),
            model=AutoRFModelConfig(
                _target=AutoRFModel,
                loss_coefficients={"rgb_loss": 1.0},
                num_coarse_samples=128,
                background_color="black",
                enable_collider=True,
                collider_params={
                    "near_plane": 0.1,
                    "far_plane": 50.0,
                },
            ),
            load_pretrain_model_path=Path('./outputs/autorf-stage-one/autorf/2023-06-16_133002/nerfstudio_models/step-000500000.ckpt')
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "latent_vectors": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000,),
            }
        },
        vis="wandb",
    ),
    description="AutoRF stage two implementations",
)

CodeNerfStageTwo = MethodSpecification(
    config=TrainerConfig(
        experiment_name="codenerf-stage-two-sv",
        use_grad_scaler=False,
        method_name="codenerf-stage-two",
        mixed_precision=False,
        max_num_iterations=50000,
        load_dir=None,
        steps_per_eval_batch=1,
        steps_per_eval_image=2000,
        steps_per_eval_all_images=49999,
        machine=MachineConfig(
            seed=114514,
            num_gpus=1,
            num_machines=1,
        ),
        pipeline=CodeNerfStageTwoPipelineConfig(
            datamanager=CarPatchDataManagerConfig(
                _target=CarPatchDataManager[CarPatchDataset],
                dataparser=KittiMotDataParserConfig(
                    train_split_fraction = 1.0,
                    sequence_num= 1,
                    start_id= 73,
                    end_id=138,                
                ),
                train_num_rays_per_batch=1024,
                train_num_images_to_sample_from=90,
                train_num_times_to_repeat_images=90,
                eval_num_images_to_sample_from=91,
                eval_num_times_to_repeat_images=91,
                n_instance_per_batch=3,
            ),
            model=CodeNerfConfig(
                _target=CodeNerfModel,
                loss_coefficients={"rgb_loss": 1.0,},
                num_coarse_samples=64,
                num_importance_samples=32,
                background_color="black",
                enable_collider=True,
                collider_params={
                    "near_plane": 0.1,
                    "far_plane": 50.0,
                },
            ),
            load_pretrain_model_path=Path('outputs/codenerf-stage-one-ko-sv/codenerf/2023-06-26_131959/nerfstudio_models/step-000499999.ckpt')
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "latent_vectors": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000,),
            }
        },
        vis="wandb",
    ),
    description="CodeNerf stage two implementations",
)