# UM-CAM
This repository provides a weakly supervised method utilizing image-level labels for medical segmentation.

If you use this toolkit, please cite the following paper:
- J. Fu, T. Lu, S. Zhang, G. Wang, UM-CAM: Uncertainty-weighted multi-resolution class activation maps for weakly-supervised fetal brain segmentation, in: MICCAI, 2023, pp. 315–324.

BibTeX entry:

    @inproceedings{fu2023cam,
    title={UM-CAM: Uncertainty-weighted Multi-resolution Class Activation Maps for Weakly-supervised Fetal Brain Segmentation},
    author={Fu, Jia and Lu, Tao and Zhang, Shaoting and Wang, Guotai},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    pages={315--324},
    year={2023},
    organization={Springer}
    }


## How to use
### 1. Raw Psuedo label based on UM-CAM
#### 1.1 Train the classification network
```python
python code_cls/train_cls.py --root_path data_dir
```

#### 1.2 Generate CAMs
```python
python code_cls/generate_cam.py --root_path data_dir --layer_num 30
```

#### 1.3 UM-CAM
```python
python code_cls/generate_umcam.py
```

### 2. Refined pseudo label based on GSE method
```python
python code_cls/GSE_refinement.py
```

### 3. Noise-robust learning based on RVC strategy
```python 
python code_seg/train_RVC.py
```


## Requirements
Before you can use this package for image segmentation. You should:
- PyTorch version >=1.12.1
- Some common python package such as Numpy, Pandas, OpenCV, scipy, SimpleITK, ......
- Intall the [FastGeodis](https://github.com/masadcv/FastGeodis) for geodesic distance transformation


## Acknowledgement and Statement
- We thank the authors of [FastGeodis](https://github.com/masadcv/FastGeodis), [MIDeepSeg](https://github.com/HiLab-git/MIDeepSeg) and [PyMIC](https://github.com/HiLab-git/PyMIC) for their elegant and efficient code base
- This project was designed for academic research, not for clinical or commercial use.
