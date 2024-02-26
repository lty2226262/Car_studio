# car-studio

### [Project Page](https://lty2226262.github.io/car-studio/)  | [Paper](https://ieeexplore.ieee.org/document/10380654) | [Data](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tliubk_connect_ust_hk/EgrO_H2uqgxPik7rP7mR-wABpwznNnV8PbV6GQxCwtwtyA?e=CZJCbU)

# News:

- [2024/02/26] Our codes have been released, feel free to explore it!
- [2024/01/02] Car-Studio is accepted to RA-L!


## Install

```bash
conda create -n car-studio python=3.8 pip -y
conda activate car-studio

git submodule update --init
cd dependencies/nerfstudio
pip install -e .

cd dependencies/segment-anything
pip install -e .

cd ../../
pip install -e .

pip uninstall nvidia-cublas-cu11
pip install nvidia-cublas-cu12
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Preparation of the dataset

### KITTI Multi-Object Tracking(MOT)

To prepare the KITTI multi-object tracking dataset, follow these steps:

1. Download KITTI multi-oject tracking dataset from `https://www.cvlibs.net/datasets/kitti/eval_tracking.php`
	- Download left color images of tracking data set (15 GB);
	- Download camera calibration matrices of tracking data set (1 MB);
	- Download training labels of tracking data set (9 MB).

2. Unzip the downloaded files and place them in the directory `./data/kitti-mot`.


### Process KITTI MOT

To convert the original KITTI multi-object tracking dataset to the CarPatch3D dataset, follow these steps:

1. Download the pre-trained model for segment-anything:

```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O dependencies/segment-anything/sam_vit_h_4b8939.pth
```

2. Run the script to process the KITTI MOT dataset:

```
python car_studio/scripts/datasets/process_kitti.py --dataset km --data_dir "./data/kitti-mot"
```

**Alternatively**, you can directly download the processed dataset 'km.zip' from the following link: [here](https://hkustconnect-my.sharepoint.com/personal/tliubk_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Ftliubk%5Fconnect%5Fust%5Fhk%2FDocuments%2Fdataset%2FCarPatch3D&ga=1)

### Process More Data(Optional)

Using the same method as for the KITTI MOT dataset, you can also process or download the [KITTI OBJ dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) and the [DVM-CAR](https://deepvisualmarketing.github.io/) dataset. Please note that processing or downloading all three datasets simultaneously is not necessary. Using the KITTI MOT dataset alone can achieve an out-of-the-box performance.

If you want to process the DVM CARS dataset, you will need additional dependencies:

```
wget https://tri-ml-public.s3.amazonaws.com/github/dd3d/experiments/4elbgev2-20210825_201852/model_final.pth

pip install hydra fvcore
pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install mpi4py
pip install seaborn numba
```

### File Tree of the Dataset

Ensure that the directory tree of the dataset is organized as follows, with the `car_studio` and `kitti-mot` folders in the same directory:

```bash
./data/
|-- car_studio
|   |-- dv.json
|   |-- km.json
|   |-- ko.json
|   |-- mask
|   `-- patch
|-- kitti-mot
|   |-- testing
|   `-- training
|-- kitti-obj
|   |-- testing
|   `-- training
`-- dvm
    `-- resized_DVM
```

## Train

Our code is based on the code architecture of nerfstudio. We recommend considering referring to the [nerfstudio tutorial](https://docs.nerf.studio/quickstart/first_nerf.html) to enhance your understanding of our code and to assist with any configuration or modifications you may wish to make.

Codes to train a car-nerf:

```bash
cd car_studio
ns-train car-nerf
```

## Test

Codes to test a car-nerf, (Please make sure to replace `<path>` in load_dir with the appropriate path to your checkpoint path.):

```bash
cd car_studio
ns-train car-nerf --max_num_iterations 10 --load_dir <outputs/.../nerfstudio_models/>  --steps_per_eval_all_
images 1
```

## Bibtex
If this work is helpful for your research, please cite the following BibTeX entry.

```
@misc{liu2023carstudio,
      title={Car-Studio: Learning Car Radiance Fields from Single-View and Endless In-the-wild Images}, 
      author={Tianyu Liu and Hao Zhao and Yang Yu and Guyue Zhou and Ming Liu},
      year={2023},
      eprint={2307.14009},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Acknowledgment

We extend our gratitude to the exceptional open-source projects:
- [KITTI](https://www.cvlibs.net/datasets/kitti/index.php)
- [DVM-CAR](https://deepvisualmarketing.github.io/)
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- [pixelnerf](https://github.com/sxyu/pixel-nerf)
- [codenerf](https://github.com/wbjang/code-nerf)
- [autorf](https://github.com/skyhehe123/AutoRF-pytorch)
- [NeuralSceneGraphs](https://github.com/princeton-computational-imaging/neural-scene-graphs)
- [mars](https://github.com/OPEN-AIR-SUN/mars)
