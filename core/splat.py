import torch
import plyfile
from utils.ply_to_ckpt import generate_gsplat_compatible_data


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
                
            means = means.to(device)
            quats = quats.to(device)
            scales = scales.to(device)
            opacities = opacities.to(device)
            colors = colors.to(device)
            norms = norms.to(device)
            quats = quats / quats.norm(dim=-1, keepdim=True)
            scales = torch.exp(scales)
            opacities = torch.sigmoid(opacities).squeeze(-1)

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