import plyfile
import numpy as np
import torch
import copy
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import math
from sklearn.decomposition import PCA

from torch.cuda.amp.grad_scaler import GradScaler
from sklearn.decomposition import PCA


def ckpt2ply(input_ckpt_file, output_ply_file="example.ply"):
    """
    Converts a Nerfstudio ckpt file (with 3D Gaussians) back into an Inria 3D Gaussians PLY file.
    
    Args:
        input_ckpt_file (str): Path to the checkpoint file containing the gauss_params dictionary.
        output_ply_file (str): Path to the output PLY file.
    """
    # ------------------
    # 1) Load checkpoint
    # ------------------
    print(f"Loading checkpoint file: {input_ckpt_file}")
    ckpt = torch.load(input_ckpt_file, map_location="cpu")
    gauss_dict = ckpt["pipeline"]  # typically where your _model.gauss_params.* keys live

    # You may need to adjust this prefix if your checkpoint naming differs
    prefix = "_model.gauss_params."

    means      = gauss_dict[prefix + "means"]      # shape: [N, 3]
    scales     = gauss_dict[prefix + "scales"]     # shape: [N, 3]
    opacities  = gauss_dict[prefix + "opacities"]  # shape: [N, 1]
    quats      = gauss_dict[prefix + "quats"]      # shape: [N, 4]
    features_dc   = gauss_dict[prefix + "features_dc"]   # shape: [N, 3]
    features_rest = gauss_dict[prefix + "features_rest"] # shape: [N, M, 3]

    # Move all to numpy (if they are still in torch Tensors)
    means         = means.cpu().numpy()
    scales        = scales.cpu().numpy()
    opacities     = opacities.cpu().numpy()
    quats         = quats.cpu().numpy()
    features_dc   = features_dc.cpu().numpy()
    features_rest = features_rest.cpu().numpy()

    N = means.shape[0]            # number of Gaussians
    M = features_rest.shape[1]    # number of "rest" SH coefficients (L^2 - 1) in each color channel

    # ----------------------------------------------------
    # 2) Construct a structured array for saving as a PLY
    # ----------------------------------------------------
    # Each 'vertex' in the .ply will have these fields
    # (mirror the property names from your `convert_ply_to_ckpt` reading code).
    # The reading code expects them in exactly this naming/ordering:

    # Basic list of fields:
    vertex_dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"), 
    ]

    # Add fields for the "rest" coefficients: f_rest_0, f_rest_1, ...
    # The reading code uses the indexing: color_index = j * num_elements + i
    # with i in [0..(M-1)] and j in [0..2]. Thus total is 3*M.
    for idx in range(3 * M):
        vertex_dtype.append((f"f_rest_{idx}", "f4"))

    # Create empty structured array
    vertex_data = np.empty(N, dtype=vertex_dtype)

    # Fill the fields
    vertex_data["x"] = means[:, 0]
    vertex_data["y"] = means[:, 1]
    vertex_data["z"] = means[:, 2]
    
    vertex_data["nx"] = np.zeros(N)
    vertex_data["ny"] = np.zeros(N)
    vertex_data["nz"] = np.zeros(N)

    vertex_data["scale_0"] = scales[:, 0]
    vertex_data["scale_1"] = scales[:, 1]
    vertex_data["scale_2"] = scales[:, 2]

    vertex_data["rot_0"] = quats[:, 0]
    vertex_data["rot_1"] = quats[:, 1]
    vertex_data["rot_2"] = quats[:, 2]
    vertex_data["rot_3"] = quats[:, 3]

    vertex_data["opacity"] = opacities[:, 0]

    vertex_data["f_dc_0"] = features_dc[:, 0]
    vertex_data["f_dc_1"] = features_dc[:, 1]
    vertex_data["f_dc_2"] = features_dc[:, 2]

    # Now fill in the "f_rest_i" fields
    # features_rest has shape [N, M, 3],
    # so for i in [0..M-1] and j in [0..2],
    # the index = j*M + i. We place features_rest[n, i, j] in f_rest_{index}.
    for i in range(M):
        for j in range(3):
            color_index = j * M + i
            vertex_data[f"f_rest_{color_index}"] = features_rest[:, i, j]

    # ---------------------------------------------------------
    # 3) Create a PlyElement and PlyData object, then save PLY
    # ---------------------------------------------------------
    el = plyfile.PlyElement.describe(vertex_data, "vertex")
    ply_data = plyfile.PlyData([el], text=True)

    print(f"Writing PLY file to: {output_ply_file}")
    ply_data.write(output_ply_file)
    print("Done!")

def convert_ply_to_ckpt(input_ply_file, 
                        input_ckpt_file,
                        output_ckpt_file="example.ckpt"):
    """
        Converts a Inria's ply file to a Nerfstudio's ckpt file.
    """
    input_ply = plyfile.PlyData.read(input_ply_file)

    # load ckpt file
    ckpt = torch.load(input_ckpt_file)


    # copy ply compatible ckpt to separate dict
    ply_compatible_ckpt = copy.deepcopy(ckpt)

    # Getting number of elements in spherical harmonics

    num_elements = len([1 for i in range(25 * 3) if f"f_rest_{i}" in input_ply["vertex"]]) // 3

    # prefix = "_model.gauss_params."


    print("Converting ply to ckpt...")
    population_candidates = generate_ply_population(input_ply_file)

    

    print("Starting to update ckpt...")

    for key in population_candidates.keys():
        print(key)

        # print(postshot_ply_state_dict["pipeline"][key].shape)
        print(ply_compatible_ckpt['pipeline'][key].shape)

        print(population_candidates[key].shape)
        ply_compatible_ckpt["pipeline"][key] = population_candidates[key]

    print("Saving ckpt...")
    torch.save(ply_compatible_ckpt, output_ckpt_file)

    print("Conversion finished.")


def generate_ply_population(input_ply_file):

    input_ply = plyfile.PlyData.read(input_ply_file)

    prefix = "_model.gauss_params."

    population_candidates = {}

    num_elements = len([1 for i in range(25 * 3) if f"f_rest_{i}" in input_ply["vertex"]]) // 3

    
    features_dc = get_features_dc(input_ply)
    features_rest = get_features_rest(input_ply, num_elements=num_elements)
    means = get_gaussian_means(input_ply)
    scales = get_gaussians_covariances(input_ply)
    opacities = get_gaussian_opacities(input_ply)
    quats = get_gaussian_rotations(input_ply)


    population_candidates[f"{prefix}features_dc"] = features_dc
    population_candidates[f"{prefix}features_rest"] = features_rest
    population_candidates[f"{prefix}means"] = means
    population_candidates[f"{prefix}scales"] = scales
    population_candidates[f"{prefix}opacities"] = opacities
    population_candidates[f"{prefix}quats"] = quats

    # Make aabb of the means
    aabb_min = means.min(dim=0).values
    aabb_max = means.max(dim=0).values

    # print aabb_min, aabb_max
    print("aabb_min", aabb_min)
    print("aabb_max", aabb_max)

    return population_candidates

def generate_gsplat_compatible_data(input_ply_file, args):
    print("=================== Reading ply file ===================")
    ply_file = plyfile.PlyData.read(input_ply_file)
    print("=================== ply file Loaded! ===================")
    num_elements = len([1 for i in range(25 * 3) if f"f_rest_{i}" in ply_file["vertex"]]) // 3

    features_dc = get_features_dc(ply_file)
    features_rest = get_features_rest(ply_file, num_elements=num_elements)
    means = get_gaussian_means(ply_file)
    scales = get_gaussians_covariances(ply_file)
    opacities = get_gaussian_opacities(ply_file)
    quats = get_gaussian_rotations(ply_file)
    

    colors = torch.cat([features_dc[:, None, :], features_rest], dim=1)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    
    if args.language_feature:
        language_feature = get_language_feature(args.language_feature)
        assert language_feature.shape[0] == means.shape[0], "Language feature and means must have the same number of elements"
        return means, quats, scales, opacities, colors, sh_degree, language_feature
    else:
        return means, quats, scales, opacities, colors, sh_degree


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
    pca = PCA(n_components=3)
    (language_feature, _) = torch.load(ckpt_file)
    language_feature = pca.fit_transform(language_feature.detach().cpu().numpy())
    language_feature = torch.tensor((language_feature - language_feature.min(axis=0)) / (language_feature.max(axis=0) - language_feature.min(axis=0)))
    print("========== Language feature loaded ==========")
    return language_feature