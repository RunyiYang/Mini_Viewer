import numpy as np
import torch 

def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def SH2RGB(sh, degree=0):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """

    input_sh = sh[:, 0, :]



    output_rgb = zeroth_order_spherical_harmonics_to_rgb(input_sh)

    # check if it's tensor
    # if type(sh) == torch.Tensor:
    #     output_rgb = output_rgb.cpu().detach().numpy()
    
    return output_rgb

def zeroth_order_spherical_harmonics_to_rgb(sh):
    """
    Convert the zeroth-order spherical harmonics coefficient to RGB values.
    
    Parameters:
    - sh: np.ndarray of shape [16, 3], spherical harmonics coefficients.
    
    Returns:
    - rgb: np.ndarray of shape [3], RGB color.
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5
