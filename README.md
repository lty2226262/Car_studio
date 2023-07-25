# Car-Studio: Learning Car Radiance Fields from Single-View and Endless In-the-wild Images
### [Project Page](https://lty2226262.github.io/car-studio/)  | [Paper]() | [Data](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tliubk_connect_ust_hk/EgrO_H2uqgxPik7rP7mR-wABpwznNnV8PbV6GQxCwtwtyA?e=CZJCbU)

The code will be released once it passes the review and is accepted.

### Directory structure
The following directory structure is required for the dataset. Please ensure that the downloaded `car_studio` and `{kitti-mot,kitti-obj}` (downloaded from the [official site](https://www.cvlibs.net/datasets/kitti/index.php)) directories are located in the same parent directory:

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
    |-- testing
    `-- training
```

### Acknowledgment

We extend our gratitude to the exceptional open-source projects:
- [KITTI](https://www.cvlibs.net/datasets/kitti/index.php)
- [DVM-CAR](https://deepvisualmarketing.github.io/)
