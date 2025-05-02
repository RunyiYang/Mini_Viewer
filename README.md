# Ply Viewer
Adapted codebase for loading Inria's ply files. To add features, you need an ckpt file to store all of the features of the ply file. 

## Install 
Python 3.10 and CUDA 12.4 are successfully tested. The main issues is that flash-attn need CUDA>12.3.

```
    micromamba create -n viewer python=3.10 -y
    micromamba activate viewer 
    pip install -r requirements.txt
    pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
    pip install git+https://github.com/huggingface/transformers@v4.49.0-SigLIP-2
    pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
    pip install flash-attn --no-build-isolation
```

### CUDA Version
CUDA toolkit 11.8 is recommended for running this renderer (check which you have with `nvcc --version` in terminal). In the INSAIT server, you could use these commands to init

```
export CC=/usr/bin/gcc-11.5
export CXX=/usr/bin/g++-11.5
export LD=/usr/bin/g++-11.5
export TORCH_CUDA_ARCH_LIST="8.6;8.9" # You need to checkout the compute capability of your device. In insait server, A6000 is 8.6, l4-24g is 8.9
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-12.4.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-12.4.0/bin:$PATH
export CPLUS_INCLUDE_PATH=/opt/modules/nvidia-cuda-12.4.0/include
```


## Example run
```
python run_viewer.py --ply splat.ply
```
### Run with Language Features
```
python run_viewer.py --ply splat.ply --language_feature langfeat.pth
```

### Language Feature architecture
1. N is the num of points in splat.ply, language feature .pth file is in ((N, D), 0). To load the feature, this code uses `language_feature, _ = torch.load(pth_file)`
2. D is 512 in default. 
3. Do not try to use the viewer to read data across the nodes. !!! Very slow

If you want to add new stuff, check out `viewer.py` `ViewerEditor` class.

# TODO: Add your needs here and we could try to implement
[x] - Object Query and Removal

[] - Object Selection

[] - Camera Placement: Placement and interpolate the path 

## Acknowledgment 

https://github.com/hangg7/nerfview - no license yet.

## Citations
If you use this library or find the repo useful for your research, please consider citing:

```
@article{wu2023mars,
  author    = {Wu, Zirui and Liu, Tianyu and Luo, Liyi and Zhong, Zhide and Chen, Jianteng and Xiao, Hongmin and Hou, Chao and Lou, Haozhe and Chen, Yuantao and Yang, Runyi and Huang, Yuxin and Ye, Xiaoyu and Yan, Zike and Shi, Yongliang and Liao, Yiyi and Zhao, Hao},
  title     = {MARS: An Instance-aware, Modular and Realistic Simulator for Autonomous Driving},
  journal   = {CICAI},
  year      = {2023},
}

@misc{yang2024spectrally,
      title={Spectrally Pruned Gaussian Fields with Neural Compensation}, 
      author={Runyi Yang and Zhenxin Zhu and Zhou Jiang and Baijun Ye and Xiaoxue Chen and Yifei Zhang and Yuantao Chen and Jian Zhao and Hao Zhao},
      year={2024},
      eprint={2405.00676},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{zheng2024gaussiangrasper,
  title={Gaussiangrasper: 3d language gaussian splatting for open-vocabulary robotic grasping},
  author={Zheng, Yuhang and Chen, Xiangyu and Zheng, Yupeng and Gu, Songen and Yang, Runyi and Jin, Bu and Li, Pengfei and Zhong, Chengliang and Wang, Zengmao and Liu, Lina and others},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}

@article{li2025scenesplat,
  title={SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining},
  author={Li, Yue and Ma, Qi and Yang, Runyi and Li, Huapeng and Ma, Mengjiao and Ren, Bin and Popovic, Nikola and Sebe, Nicu and Konukoglu, Ender and Gevers, Theo and others},
  journal={arXiv preprint arXiv:2503.18052},
  year={2025}
}
```