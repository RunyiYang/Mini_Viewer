import plyfile
import numpy as np
import torch

import math
from sklearn.decomposition import PCA


def generate_gsplat_compatible_data(input_ply_file, args):
    print("=================== Reading ply file ===================")
    ply_file = plyfile.PlyData.read(input_ply_file)
    print("=================== ply file Loaded! ===================")
    num_elements = len([1 for i in range(25 * 3) if f"f_rest_{i}" in ply_file["vertex"]]) // 3

    features_dc = get_features_dc(ply_file)
    features_rest = get_features_rest(ply_file, num_elements=num_elements)
    means = get_gaussian_means(ply_file)
    norms = get_gaussian_norms(ply_file)
    scales = get_gaussians_covariances(ply_file)
    opacities = get_gaussian_opacities(ply_file)
    quats = get_gaussian_rotations(ply_file)
    

    colors = torch.cat([features_dc[:, None, :], features_rest], dim=1)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    
    if args.language_feature:
        language_feature, pca = get_language_feature(args.language_feature)
        assert language_feature.shape[0] == means.shape[0], f"Language feature and means must have the same number of elements, {language_feature.shape[0], means.shape[0]}"
        return means, norms, quats, scales, opacities, colors, sh_degree, language_feature, pca
    else:
        return means, norms, quats, scales, opacities, colors, sh_degree


def get_features_dc(input_ply):
    """
    Extracts the spherical harmonic features from the Inria's input ply file.
    """

    print("Converting spherical harmonic features...")

    features_dc = []

    for i in range(3):
        features_dc.append(input_ply["vertex"][f"f_dc_{i}"])

    features_dc = np.stack(features_dc, axis=-1)

    features_dc = torch.tensor(features_dc, dtype=torch.float32)


    return features_dc


def get_features_rest(input_ply, num_elements):
    """
    Extracts the fc_rest features from the Inria's input ply file.
    """

    print("Converting fc_rest features...")

    features_rest = []

    for i in range(num_elements):

        f_rest_i = []

        for j in range(3):
            color_index = j * num_elements + i

            f_rest_i.append(input_ply["vertex"][f"f_rest_{color_index}"])

        f_rest_i = np.stack(f_rest_i, axis=-1)


        features_rest.append(f_rest_i)

    features_rest = np.stack(features_rest, axis=1)

    features_rest = torch.tensor(features_rest, dtype=torch.float32)

    return features_rest


def get_gaussian_means(input_ply):
    """
    Extracts the gaussian means from the Inria's input ply file.
    """
    axes = ["x", "y", "z"]

    means = []
    for i, axis in enumerate(axes):
        means.append(input_ply["vertex"][axis])

    means = np.stack(means, axis=-1)

    means = torch.tensor(means, dtype=torch.float32)

    return means

def get_gaussian_norms(input_ply):
    """
    Extracts the gaussian means (normal vectors) from the Inria's input ply file.
    If a normal component ('nx', 'ny', or 'nz') is missing, it is replaced with zeros.
    """
    axes = ["nx", "ny", "nz"]

    # Determine the number of vertices from an existing attribute (e.g., "x")
    num_vertices = input_ply["vertex"]["x"].shape[0] if hasattr(input_ply["vertex"]["x"], "shape") else len(input_ply["vertex"]["x"])

    norms = []
    for axis in axes:
        # Use the normal component if it exists; otherwise, create a zero array
        if axis in input_ply["vertex"]:
            norms.append(input_ply["vertex"][axis])
        else:
            norms.append(np.zeros(num_vertices))
            
    norms = np.stack(norms, axis=-1)
    norms = torch.tensor(norms, dtype=torch.float32)
    return norms


def get_gaussians_covariances(input_ply):
    """
    Extracts the gaussian covariances from the Inria's input ply file.
    """
    axes = ["0", "1", "2"]

    scales = []
    for _, axis in enumerate(axes):
        scales.append(input_ply["vertex"][f"scale_{axis}"])

    scales = np.stack(scales, axis=-1)

    scales = torch.tensor(scales, dtype=torch.float32)

    return scales

def get_gaussian_opacities(input_ply):
    """
    Extracts the gaussian opacities from the Inria's input ply file.
    """
    opacities = input_ply["vertex"]["opacity"]

    opacities = torch.tensor(opacities, dtype=torch.float32).unsqueeze(-1)

    return opacities

def get_gaussian_rotations(input_ply):
    """
    Extracts the gaussian rotations (in wxyz form) from the Inria's input ply file.
    """
    quats = []

    for i in range(4):
        quats.append(input_ply["vertex"][f"rot_{i}"])

    quats = np.stack(quats, axis=-1)

    quats = torch.tensor(quats, dtype=torch.float32)

    return quats

def get_language_feature(ckpt_file):
    """
    Extracts the language feature from the Inria's input ply file.
    """
    print("========== Loading language feature ==========")
    if ckpt_file.endswith(".npy"):
        language_feature_large = np.load(ckpt_file)
    elif ckpt_file.endswith(".pth"):
        try: 
            (language_feature_large, _) = torch.load(ckpt_file)
        except:
            language_feature_large = torch.load(ckpt_file)
        language_feature_large = language_feature_large.detach().cpu().numpy()
    else:
        raise ValueError("Unsupported file format. Please provide a .npy or .pth file.")
    pca = PCA(n_components=3)
    language_feature = pca.fit_transform(language_feature_large)
    language_feature = torch.tensor((language_feature - language_feature.min(axis=0)) / (language_feature.max(axis=0) - language_feature.min(axis=0)))
    print("========== Language feature loaded ==========")
    return language_feature, language_feature_large