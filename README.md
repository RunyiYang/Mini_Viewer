# Ply Viewer
Adapted codebase for loading Inria's ply files. To add features, you need an ckpt file to store all of the features of the ply file. 

## Install 

```
    micromamba create -n python=3.10 -y
    micromamba activate python=3.10
    pip install -r requirements.txt
    pip install torch torchvision
```

## Example run
```
python run_viewer.py --ply splat.ply
```


If you want to add new stuff, check out `viewer.py` `ViewerEditor` class.


## Acknowledgment 

https://github.com/hangg7/nerfview - no license yet.
