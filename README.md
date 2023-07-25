# Car-Studio: Learning Car Radiance Fields from Single-View and Endless In-the-wild Images
### [Project Page](https://lty2226262.github.io/car-studio/)  | [Paper]() | [Data](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tliubk_connect_ust_hk/EgrO_H2uqgxPik7rP7mR-wABpwznNnV8PbV6GQxCwtwtyA?e=CZJCbU)

The code will be released once it passes the review and is accepted.

### Directory structure
The following directory structure is required for the dataset. Please ensure that the downloaded `car_studio`,`{kitti-mot,kitti-obj}` (download from the [official site](https://www.cvlibs.net/datasets/kitti/index.php)) or `dvm` (download from the [official site](https://deepvisualmarketing.github.io/)) directories are located in the same parent directory. Note that the `kitti-mot`, `kitti-obj`, and `dvm` datasets can be used individually and do not need to be downloaded simultaneously.

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
```

### Acknowledgment

We extend our gratitude to the exceptional open-source projects:
- [KITTI](https://www.cvlibs.net/datasets/kitti/index.php)
- [DVM-CAR](https://deepvisualmarketing.github.io/)
