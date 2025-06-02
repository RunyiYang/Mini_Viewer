import numpy as np
import cv2
import torch

def apply_bilateral_filter(depth_map, d=9, sigma_color=75, sigma_space=75):
    # Apply bilateral filter
    filtered_depth_map = cv2.bilateralFilter(depth_map.astype(np.float32), d, sigma_color, sigma_space)
    return filtered_depth_map

def apply_histogram_equalization(depth_map):
    # Normalize depth map to 0-255 range
    depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Apply histogram equalization
    equalized_depth_map = cv2.equalizeHist(depth_map_norm)
    return equalized_depth_map

def apply_unsharp_mask(depth_map, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(depth_map, (0, 0), sigma)
    sharpened = cv2.addWeighted(depth_map, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def depth_to_rgb(
    depth_map: np.ndarray,
    *,
    depth_scale: float = 1.0,          # raw units → metres
    min_depth: float | None = None,    # clip range; None = robust percentiles
    max_depth: float | None = None,
    max_valid_depth: float | None = None,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Visualise a single-channel depth map with a perceptually uniform colour ramp.

    Parameters
    ----------
    depth_map : H×W np.ndarray  (uint16/uint32/float)
    depth_scale : float
        How to turn the raw sensor units into metres.
        e.g. RealSense D435:  depth_scale = 0.001
    min_depth, max_depth : float | None
        Range to colourise, in metres.  If either is None we use the
        2-nd and 98-th percentiles of valid pixels for a robust auto-stretch.
    max_valid_depth : float | None
        Any pixel deeper than this is shown as black (invalid).
    colormap : int
    """
    # ------------- metre conversion & validity mask -------------
    depth_m = depth_map.astype(np.float32) * depth_scale
    valid   = depth_m > 0
    if max_valid_depth is not None:
        valid &= depth_m <= max_valid_depth

    if not np.any(valid):
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)   # all invalid –> black

    # ------------- robust clipping range -------------
    if min_depth is None:
        min_depth = np.percentile(depth_m[valid], 2)             # near
    if max_depth is None:
        max_depth = np.percentile(depth_m[valid], 98)            # far
    if max_depth <= min_depth:                                   # degenerate case
        max_depth = min_depth + 1e-6

    # ------------- normalise to 0…255 & colour map -------------
    depth_clipped = np.clip(depth_m, min_depth, max_depth)
    norm = (depth_clipped - min_depth) / (max_depth - min_depth)   # 0 (near)…1 (far)
    norm = 1.0 - norm                                              # invert – near = hot
    norm_u8 = (norm * 255).astype(np.uint8)

    vis = cv2.applyColorMap(norm_u8, colormap)
    vis[~valid] = 255                                               # white
    return vis


def depth_to_normal(
    depth_map: np.ndarray,
    *,                         # ← only keyword args below this line
    K: np.ndarray,          # camera intrinsics (3×3)
    hsv: bool = True,          # False → classic tangent-space (purple-ish)
    strength: float = 0.75,     # exaggerate slopes (just like before)
    smooth_ksize: int = 10      # bilateral filter to tame speckle
) -> np.ndarray:
    """
    Camera-space or HSV-encoded normal map from a linear-depth image.

    Parameters
    ----------
    depth_map : H×W depth (uint16, uint32, float…)
    fx, fy, cx, cy : float
        Camera intrinsics (pinhole model, same units as pixels in depth_map).
    hsv : bool
        True  → colourful HSV encoding (recommended for visualisation)  
        False → standard tangent-space (R,G,B) = (X,Y,Z) for engines.
    strength : float
        Multiplies the X/Y slopes before renormalising.  Values >1 make
        gentle undulations more obvious.
    smooth_ksize : int
        Diameter of the bilateral filter.  0 → no smoothing.
    """
    valid = depth_map > 0
    if not np.any(valid):
        return np.ones((*depth_map.shape, 3), dtype=np.uint8)
    K = K.cpu().squeeze().numpy() if isinstance(K, torch.Tensor) else K
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z = depth_map.astype(np.float32)
    if smooth_ksize >= 3:
        # bilateral keeps edges sharper than Gaussian
        Z = cv2.bilateralFilter(Z, smooth_ksize, 0.1, 5)

    h, w = Z.shape

    # -- back-project each pixel to camera space ---------------------------
    # x_cam = (u - cx) * Z / fx ,  y_cam = (v - cy) * Z / fy ,  z_cam = Z
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    X = (us - cx) * Z / fx
    Y = (vs - cy) * Z / fy

    # -- neighbour vectors -------------------------------------------------
    # shift –X (right neighbour minus self)
    v_x = np.dstack((np.roll(X, -1, axis=1) - X,
                     np.roll(Y, -1, axis=1) - Y,
                     np.roll(Z, -1, axis=1) - Z))

    # shift –Y (down neighbour minus self)
    v_y = np.dstack((np.roll(X, -1, axis=0) - X,
                     np.roll(Y, -1, axis=0) - Y,
                     np.roll(Z, -1, axis=0) - Z))

    # cross-product v_x × v_y  → outward-pointing normal
    N = np.cross(v_x, v_y)
    N[..., :2] *= strength                       # optional exaggeration
    N_norm = np.linalg.norm(N, axis=2, keepdims=True) + 1e-8
    N /= N_norm                                 # unit length

    if not hsv:
        # classic tangent-space: pack [-1,1] → [0,255]
        normal_u8 = ((N + 1.0) * 127.5).astype(np.uint8)
        return normal_u8

    # ---------------------------------------------------------------------
    #  HSV visualisation: hue = azimuth, sat = tilt, val = 1
    # ---------------------------------------------------------------------
    nx, ny, nz = N[..., 0], N[..., 1], N[..., 2]

    # azimuth: 0..2π → 0..179 (OpenCV 8-bit HSV hue range)
    hue = (np.arctan2(ny, nx) + np.pi) * (179 / (2 * np.pi))

    # tilt: 0 (front-facing) … 1 (side-on) → saturation 0..255
    sat = np.clip(np.sqrt(1.0 - nz), 0, 1) * 255

    hsv_img = np.dstack((hue, sat, np.full_like(hue, 255))).astype(np.uint8)
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    # seg invalid pixels to white
    rgb_img[~valid] = 255

    return rgb_img


def smooth_depth_map(depth_map, ksize=5, sigma=2):
    # Apply Gaussian blur to smooth the depth map
    smoothed_depth_map = cv2.GaussianBlur(depth_map, (ksize, ksize), sigma)
    return smoothed_depth_map