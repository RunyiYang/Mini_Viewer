# Ply Viewer
Adapted codebase for loading Inria's ply files. To add features, you need an ckpt file to store all of the features of the ply file. 

## Install 
Python 3.10 and CUDA 11.8 are recommended for this package.

```
    micromamba create -n viewer python=3.10 -y
    micromamba activate viewer 
    pip install -r requirements.txt
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

### CUDA Version
CUDA toolkit 11.8 is recommended for running this renderer (check which you have with `nvcc --version` in terminal). In the INSAIT server, you could use these commands to init

```
export CC=/usr/bin/gcc-11.5
export CXX=/usr/bin/g++-11.5
export LD=/usr/bin/g++-11.5
export TORCH_CUDA_ARCH_LIST="8.6;8.9" # You need to checkout the compute capability of your device. In insait server, A6000 is 8.6, l4-24g is 8.9
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8.0/bin:$PATH
export CPLUS_INCLUDE_PATH=/opt/modules/nvidia-cuda-11.8.0/include
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
