import torch
import plyfile
from utils.ply_to_ckpt import generate_gsplat_compatible_data
import numpy as np
import os
from sklearn.decomposition import PCA
from utils.color_shs import RGB2SH, SH2RGB

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
                language_feature = torch.tensor(language_feature).to(device).to(torch.float32)
                language_feature_large = torch.tensor(language_feature_large).to(device).to(torch.float32)
                colors = SH2RGB(colors, sh_degree)
                sh_degree = None
            else:
                means, norms, quats, scales, opacities, colors, sh_degree = gaussian_params
            if args.prune:
                # masks = torch.from_numpy(np.load(os.path.join(args.folder_npy, 'valid_feat_mask.npy'))).bool()
                mask = (scales < 0.5).all(dim=-1) & (opacities > 0.8).all(dim=-1)
                print(f"There are {mask.sum()} valid splats out of {means.shape[0]}")
            
            else:
                mask = torch.ones(means.shape[0], dtype=torch.bool)
            
            means = means[mask]
            norms = norms[mask]
            quats = quats[mask]
            scales = scales[mask]
            opacities = opacities[mask]
            colors = colors[mask]
            
            if args.language_feature:
                language_feature = language_feature[mask]
                language_feature_large = language_feature_large[mask]
                
            
            quats = quats / quats.norm(dim=-1, keepdim=True)
            scales = torch.exp(scales)
            opacities = torch.sigmoid(opacities).squeeze(-1)
        
        
        
        if args.folder_npy is not None:
            # Load data as before
                        # Optional: Prune entries where any dimension of scale is >= 1
            
            means = torch.from_numpy(np.load(os.path.join(args.folder_npy, 'coord.npy'))).float()
            if os.path.exists(os.path.join(args.folder_npy, 'normal.npy')):
                norms = torch.from_numpy(np.load(os.path.join(args.folder_npy, 'normal.npy'))).float()
            else:
                norms = torch.zeros(means.shape)
            quats = torch.from_numpy(np.load(os.path.join(args.folder_npy, 'quat.npy'))).float()
            scales = torch.from_numpy(np.load(os.path.join(args.folder_npy, 'scale.npy'))).float()
            opacities = torch.from_numpy(np.load(os.path.join(args.folder_npy, 'opacity.npy'))).float()
            colors = torch.from_numpy(np.load(os.path.join(args.folder_npy, 'color.npy'))).float() / 255.0
            
            self.save_params_histograms(means, scales, colors, opacities)
            if args.prune:
                masks = torch.from_numpy(np.load(os.path.join(args.folder_npy, 'valid_feat_mask.npy'))).bool()
                mask =  (scales < 0.5).all(dim=-1) & masks & (means[:, 2] < 2.0)
                # 
                print(f"There are {mask.sum()} valid splats out of {means.shape[0]}")
            else:
                mask = torch.ones(means.shape[0], dtype=torch.bool)
            sh_degree = None
            
            means = means[mask]
            norms = norms[mask]
            quats = quats[mask]
            scales = scales[mask]
            opacities = opacities[mask]
            colors = colors[mask]
            if args.language_feature:
                if ".pth" in args.language_feature:
                    language_feature_large = torch.load(os.path.join(args.folder_npy, args.language_feature))[mask].detach().to("cpu").numpy()
                    print(language_feature_large.shape)
                else:
                    language_feature_large = np.load(os.path.join(args.folder_npy, args.language_feature)+'.npy')[mask.numpy()]
                pca = PCA(n_components=3)
                language_feature = pca.fit_transform(language_feature_large)
                language_feature = torch.tensor((language_feature - language_feature.min(axis=0)) / (language_feature.max(axis=0) - language_feature.min(axis=0))).to(torch.float).to(device)
                language_feature_large = torch.tensor(language_feature_large).to(torch.float).to(device)

        # Get me the all params histogram and save to the data folder
        # self.save_params_histograms(means, scales, colors, opacities)
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
    
    

    def save_params_histograms(self, means, scales, colors, opacities, save_dir="./datastats"):
        """
        Save histograms for means (x, y, z), scales (x, y, z),
        colors (r, g, b), and opacities (if desired) in the given folder.
        Norms and quats are not analyzed.
        """
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert tensors to CPU numpy arrays (if not already)
        means_np = means.cpu().numpy()  # shape (N,3)
        scales_np = scales.cpu().numpy()  # shape (N,3)
        colors_np = colors.cpu().numpy()  # shape (N,3), expected normalized [0,1] or [0,255]
        opacities_np = opacities.cpu().numpy().flatten()  # 1D array
        
        # --- Means histograms ---
        dims = ['x', 'y', 'z']
        plt.figure(figsize=(15, 5))
        for i, axis in enumerate(dims):
            plt.subplot(1, 3, i+1)
            plt.hist(means_np[:, i], bins=50, color='blue', alpha=0.7, edgecolor='black')
            plt.title(f"Means {axis}-axis")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
        means_path = os.path.join(save_dir, "means_histogram.png")
        plt.tight_layout()
        plt.savefig(means_path)
        plt.close()
        print(f"Saved means histogram to {means_path}")
        
        # --- Scales histograms ---
        plt.figure(figsize=(15, 5))
        for i, axis in enumerate(dims):
            plt.subplot(1, 3, i+1)
            plt.hist(scales_np[:, i], bins=50, color='green', alpha=0.7, edgecolor='black')
            plt.title(f"Scales {axis}-axis")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
        scales_path = os.path.join(save_dir, "scales_histogram.png")
        plt.tight_layout()
        plt.savefig(scales_path)
        plt.close()
        print(f"Saved scales histogram to {scales_path}")
        
        # --- Colors histograms (R, G, B) ---
        channels = ['R', 'G', 'B']
        # Use predefined colors for plotting channels
        channel_colors = ['red', 'green', 'blue']
        plt.figure(figsize=(15, 5))
        for i, ch in enumerate(channels):
            plt.subplot(1, 3, i+1)
            plt.hist(colors_np[:, i], bins=50, color=channel_colors[i], alpha=0.7, edgecolor='black')
            plt.title(f"Colors {ch}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
        colors_path = os.path.join(save_dir, "colors_histogram.png")
        plt.tight_layout()
        plt.savefig(colors_path)
        plt.close()
        print(f"Saved colors histogram to {colors_path}")
        
        # --- Opacities histogram (optional) ---
        plt.figure(figsize=(8, 6))
        plt.hist(opacities_np, bins=50, color='purple', alpha=0.7, edgecolor='black')
        plt.title("Opacities Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        opacities_path = os.path.join(save_dir, "opacities_histogram.png")
        plt.tight_layout()
        plt.savefig(opacities_path)
        plt.close()
        print(f"Saved opacities histogram to {opacities_path}")
