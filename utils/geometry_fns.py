import numpy as np
import cv2

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


def depth_to_rgb(depth_map):
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    
    # Normalize the depth map to [0, 1]
    normalized_depth_map = (depth_map - min_depth) / (max_depth - min_depth)
    
    # Scale to [0, 255] and convert to integers
    grayscale_map = (normalized_depth_map * 255).astype(np.uint8)
    
    # Create an RGB image by stacking the grayscale map into 3 channels
    rgb_image = np.stack([grayscale_map] * 3, axis=-1)
    return rgb_image

def depth_to_normal(depth_map):
    # Compute gradients in x and y directions
    grad_y, grad_x = np.gradient(depth_map)

    # Compute the normal for each pixel
    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = np.ones_like(depth_map)

    # Normalize the normals
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm

    # Convert to RGB format: map [-1, 1] to [0, 255]
    normal_map = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
    normal_map[..., 0] = ((normal_x + 1) * 0.5 * 255).astype(np.uint8)  # Red channel
    normal_map[..., 1] = ((normal_y + 1) * 0.5 * 255).astype(np.uint8)  # Green channel
    normal_map[..., 2] = ((normal_z + 1) * 0.5 * 255).astype(np.uint8)  # Blue channel

    return normal_map

def smooth_depth_map(depth_map, ksize=5, sigma=2):
    # Apply Gaussian blur to smooth the depth map
    smoothed_depth_map = cv2.GaussianBlur(depth_map, (ksize, ksize), sigma)
    return smoothed_depth_map