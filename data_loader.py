
import torch 
import plyfile
from utils.ply_to_ckpt import generate_gsplat_compatible_data, get_language_feature


def load_data(args):
    """
        Function for loading GSplat representation data.
        If ckpt is provided, it loads the data from the checkpoint file.
        If ply is provided, it generates the data from the ply file.

        Returns data needed for rendering on our viewer.
        
        Returns:
            means: torch.Tensor, [N, 3]
            quats: torch.Tensor, [N, 4]
            scales: torch.Tensor, [N, 3]
            opacities: torch.Tensor, [N]
            colors: torch.Tensor, [N, 3]
            sh_degree: int
    """

    device = args.device

    if args.ply is not None:
        gaussian_params = generate_gsplat_compatible_data(args.ply, args)
        if args.language_feature:
            means, quats, scales, opacities, colors, sh_degree, language_feature = gaussian_params
            
            language_feature = language_feature.to(device)
        else:
            means, quats, scales, opacities, colors, sh_degree = gaussian_params
            
        means = means.to(device)
        quats = quats.to(device)
        scales = scales.to(device)
        opacities = opacities.to(device)
        colors = colors.to(device)

        quats = quats / quats.norm(dim=-1, keepdim=True)
        scales = torch.exp(scales)
        opacities = torch.sigmoid(opacities).squeeze(-1)

    if args.language_feature:
        return means, quats, scales, opacities, colors, sh_degree, language_feature
    else:
        return means, quats, scales, opacities, colors, sh_degree