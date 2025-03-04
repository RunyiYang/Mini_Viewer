import torch
import plyfile
from utils.ply_to_ckpt import generate_gsplat_compatible_data
import numpy as np
import os
from sklearn.decomposition import PCA

class SplatData:
    def __init__(self, args=None):
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._language_feature = torch.empty(0)
        self.language_feature_large = torch.empty(0)
        self.device = args.device
        
        if args:
            self.load_data(args)
    
    def load_data(self, args):
        device = self.device

        if args.ply is not None:
            gaussian_params = generate_gsplat_compatible_data(args.ply, args)
            if args.language_feature:
                means, norms, quats, scales, opacities, colors, sh_degree, language_feature, language_feature_large = gaussian_params
                language_feature = language_feature.to(device)
                language_feature_large = language_feature_large.to(device)
            else:
                means, norms, quats, scales, opacities, colors, sh_degree = gaussian_params
            
            
            quats = quats / quats.norm(dim=-1, keepdim=True)
            scales = torch.exp(scales)
            opacities = torch.sigmoid(opacities).squeeze(-1)
        
        
        
        if args.folder_npy is not None:
            # Load data as before
                        # Optional: Prune entries where any dimension of scale is >= 1

            means = torch.from_numpy(np.load(os.path.join(args.folder_npy, 'coord.npy'))).float()
            norms = torch.from_numpy(np.load(os.path.join(args.folder_npy, 'normal.npy'))).float()
            quats = torch.from_numpy(np.load(os.path.join(args.folder_npy, 'quat.npy'))).float()
            scales = torch.from_numpy(np.load(os.path.join(args.folder_npy, 'scale.npy'))).float()
            opacities = torch.from_numpy(np.load(os.path.join(args.folder_npy, 'opacity.npy'))).float()
            colors = torch.from_numpy(np.load(os.path.join(args.folder_npy, 'color.npy'))).float() / 255.0
            sh_degree = None
            if args.language_feature:
                language_feature_large = np.load(os.path.join(args.folder_npy, args.language_feature)+'.npy')
                pca = PCA(n_components=3)
                language_feature = pca.fit_transform(language_feature_large)
                language_feature = torch.tensor((language_feature - language_feature.min(axis=0)) / (language_feature.max(axis=0) - language_feature.min(axis=0))).to(device)



        means = means.to(device)
        quats = quats.to(device)
        scales = scales.to(device)
        opacities = opacities.to(device)
        colors = colors.to(device)
        norms = norms.to(device)
        quats = quats.to(device)
        scales = scales.to(device)
        opacities = opacities.to(device)
        

        if args.language_feature:
            self._means = means
            self._norms = norms
            self._quats = quats
            self._scales = scales
            self._opacities = opacities
            self._colors = colors
            self._sh_degree = sh_degree
            self._language_feature = language_feature
            self.language_feature_large = language_feature_large
        else:
            self._means = means
            self._norms = norms
            self._quats = quats
            self._scales = scales
            self._opacities = opacities
            self._colors = colors
            self._sh_degree = sh_degree
    
    def get_data(self):
        if self._language_feature is not None:
            splat_data = {
                'means': self._means,
                'norms': self._norms,
                'quats': self._quats,
                'scales': self._scales,
                'opacities': self._opacities,
                'colors': self._colors,
                'sh_degree': self._sh_degree,
                'language_feature': self._language_feature
            }
        else:
            splat_data = {
                'means': self._means,
                'norms': self._norms,
                'quats': self._quats,
                'scales': self._scales,
                'opacities': self._opacities,
                'colors': self._colors,
                'sh_degree': self._sh_degree
            }
        return splat_data
    
    
    def get_large(self):
        return self.language_feature_large