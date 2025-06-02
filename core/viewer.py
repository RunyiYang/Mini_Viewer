import torch
import torch.nn as nn
import torch.nn.functional as F
import nerfview
import functools
from sklearn.cluster import KMeans
from plyfile import PlyData, PlyElement  # PlyFile import
import numpy as np
import pdb
import os, copy
import pdb
import viser
from core.splat import SplatData
from PIL import Image
from typing import TYPE_CHECKING, Literal, Optional, Tuple, get_args
import dataclasses
import json

from actions.base import BasicFeature
from actions.language_feature import LanguageFeature

RenderAction = Literal["rerender", "move", "static", "update"]
@dataclasses.dataclass
class RenderTask(object):
    action: RenderAction
    camera_state: Optional["CameraState"] = None

class ViewerEditor(nerfview.Viewer):

    def __init__(self, 
                 splat_args, 
                 splat_data, 
                 *args, **kwargs):
        self.scene_list = kwargs.pop("scene_list", None)
        
        super().__init__(*args, **kwargs)
        self._editor = None
        self.splat_args = splat_args
        self.device = splat_args.device
        self.language_feature = splat_args.language_feature
        self.mode = "rgb"
        self.splats = splat_data
        self.adjust_viewer()
        if os.path.exists(".tmp/camera_state.json"):
            with open(".tmp/camera_state.json", "r") as f:
                camera_state_dict = json.load(f)
            self.camera_state = nerfview.CameraState(
                c2w=np.array(camera_state_dict["c2w"]),
                fov=camera_state_dict["fov"],
                aspect=camera_state_dict["aspect"],
            )

        
    def adjust_viewer(self):
        """
        Adjust the inherited viewer.
        """
        if os.path.exists(".tmp/camera_state.json"):
            with open(".tmp/camera_state.json", "r") as f:
                camera_state_dict = json.load(f)
            self.camera_state = nerfview.CameraState(
                c2w=np.array(camera_state_dict["c2w"]),
                fov=camera_state_dict["fov"] / np.pi * 180,
                aspect=camera_state_dict["aspect"],
            )
  
        with self._rendering_folder:
            self._max_img_res_slider.remove()
            self._max_img_res_slider = self.server.gui.add_slider(
                "Image Res", min=64, max=2048, step=1, initial_value=1920
            )
            self._max_img_res_slider.on_update(self.rerender)
            self._fov_slider = self.server.gui.add_slider(
                "FOV", min=10, max=120, step=1, initial_value=self.camera_state.fov if os.path.exists(".tmp/camera_state.json") else 45
            )
            self._fov_slider.on_update(self.rerender_K)
        
        # ------------------------------------------------------------------
        # Scene selector GUI
        # ------------------------------------------------------------------
        if self.scene_list:
            with self.server.gui.add_folder(label="Scene Selection"):
                self._scene_dropdown = self.server.gui.add_dropdown(
                    "Scene",
                    options=self.scene_list,
                    initial_value=self._current_scene(),
                )
                self._scene_dropdown.on_update(self._on_scene_change)
            
        @self.server.on_client_connect
        def handle_client_connect(client: viser.ClientHandle):
            print("Client connected")
            # if client.client_id not in self.processed_clients:
            self.splat_central_point = self.find_splat_central_point()
            # client.camera.position = np.array([0.0, 0.0, -10.0])
            client.camera.position = self.splat_central_point + np.array([0.0, 0.0, -10.0])
            client.camera.wxyz = np.array([0.0, 0.0, 0.0, 1.0])
            if os.path.exists(".tmp/camera_state.json"):
                client.camera.position = self.camera_state.c2w[:3, 3]
                R = self.camera_state.c2w[:3, :3]
                client.camera.wxyz = rotation_matrix_to_quaternion(R)

    # ------------------------------------------------------------------ #
    # helpers for scene switching
    # ------------------------------------------------------------------ #
    def _current_scene(self) -> str:
        """
        Return the scene id (the last folder name) of the scene
        currently loaded in `self.splat_args.folder_npy`.
        """
        if self.splat_args.folder_npy:
            return os.path.basename(self.splat_args.folder_npy.rstrip("/"))
        if self.splat_args.ply:  # fall-back for --ply mode
            # note, we assume the format: --ply gaussian_world/scannetpp_v2_val_mcmc_3dgs/09c1414f1b/ckpts/point_cloud_30000.ply
            return os.path.basename(os.path.dirname(os.path.dirname(self.splat_args.ply)))
        return ""
    
    def _on_scene_change(self, widget):
        new_scene = widget.target.value
        if new_scene != self._current_scene():
            self._switch_scene(new_scene)
    
    def _switch_scene(self, scene_id: str):
        """
        1.  constructs a new `folder_npy` by swapping the last path element
        2.  reloads the Gaussian data
        3.  hot-swaps all viewer state
        """
        if self.splat_args.folder_npy is None:
            print("[Scene switch] Only supported when viewer was started "
                "with --folder_npy.  Ignoring request.")
            return
    
        base_dir = os.path.dirname(self.splat_args.folder_npy.rstrip("/"))
        new_folder = os.path.join(base_dir, scene_id)
        print(f"[Scene switch] Switching to {new_folder}...")
    
        if not os.path.isdir(new_folder):
            print(f"[Scene switch] Folder not found: {new_folder}")
            return
    
        # ------------------------------------------------------------------
        # 1. build a *fresh* args object & load new splats
        # ------------------------------------------------------------------
        new_args = copy.deepcopy(self.splat_args)
        new_args.folder_npy = new_folder
    
        new_splats = SplatData(args=new_args)
        del self.splats, self.language_feature
    
        # ------------------------------------------------------------------
        # 2. swap state inside the running viewer
        # ------------------------------------------------------------------
        self.splat_args = new_args
        self.splats = new_splats
    
        # renderer
        self.update_splat_renderer(splats=new_splats, update_inplace_memory=False)
    
        # # feature side-panels
        # self.base_feature = BasicFeature(self, new_splats)
        # self.language_feature = LanguageFeature(self, new_splats,
        #                                         feature_type=self.splat_args.feature_type)

        # tell the *existing* panels to look at the new scene
        if hasattr(self, "base_feature_panel") and self.base_feature_panel is not None:
            self.base_feature_panel.update_splats(new_splats)
            print("[Scene switch] Updated base feature panel")
        if hasattr(self, "language_feature_panel") and self.language_feature_panel is not None:
            self.language_feature_panel.update_splats(new_splats)
            print("[Scene switch] Updated language feature panel")
    
        # move camera to the new scene’s centre
        centre = self.find_splat_central_point()

        print("[Scene switch] Finished switching to scene:", scene_id)

    def find_splat_central_point(self) -> np.ndarray:
        """
            Find the central point of the splat data.
        """
        means = self.splats._means
        return torch.mean(means, 0).cpu().numpy()

    def update_splat_renderer(self, 
                                    splats: SplatData = None,
                                    sh_degree: int = 3,
                                    backend: str = 'gsplat', 
                                    update_inplace_memory=False,
                                    render_mode="rgb"):
        """
            Update the splat renderer with the new data.

            Args:
                splat_data (SplatData): The splat data to be rendered.
                sh_degree (int): The degree of the spherical harmonics.
                device (str): The device to run the renderer on.
                backend (str): The backend to use for rendering.
                update_inplace_memory (bool): Whether to update the renderer's memory inplace.
        """
        splat_data = splats.get_data()
        
        if update_inplace_memory:
            # if means is not None:
            if splat_data["means"] is not None:
                self.splats._means = splat_data["means"]
            # if quats is not None:
            if splat_data["quats"] is not None:
                self.splats._quats = splat_data["quats"]
            # if scales is not None:
            if splat_data["scales"] is not None:
                self.splats._scales = splat_data["scales"]
            # if opacities is not None:
            if splat_data["opacities"] is not None:
                self.splats._opacities = splat_data["opacities"]
            # if colors is not None:
            if splat_data["colors"] is not None:
                self.splats._colors = splat_data["colors"]
            # if sh_degree is not None:
            if splat_data["sh_degree"] is not None:
                self.splats._sh_degree = splat_data["sh_degree"]
            if splat_data["language_feature"] is not None:
                self.splats._language_features = splat_data["language_feature"]


        render_fn = functools.partial(self.render_fn, 
                                                        means= splat_data["means"],
                                                        quats= splat_data["quats"],
                                                        # norms= splat_data["norms"],
                                                        scales = splat_data["scales"],
                                                        opacities = splat_data["opacities"],
                                                        colors = splat_data["colors"],
                                                        sh_degree = splat_data["sh_degree"],
                                                        device = self.device,
                                                        backend = backend,
                                                        render_mode = render_mode)


        self.render_fn = render_fn


        self.rerender(None) 


    def call_renderer(self, client: viser.ClientHandle):
        """
            Call the renderer function to render the scene. Exploiting nerfview._renderer.py function for this purpose.
        """
        # pdb.set_trace()
        camera_state = self.get_camera_state(client)

        img_wh = self._renderers[client.client_id]._get_img_wh(camera_state.aspect) 

        render_rgbs = self.render_fn(camera_state,
                                            img_wh)
        
        return render_rgbs
    
    def snapshot(self, idx):
        """
        Take a snapshot of the scene, replace the black background with white,
        and save the image.
        """
        client = self.server.get_clients()[0]
        render_rgbs = self.call_renderer(client)
        camera_state = self.get_camera_state(client)
        camera_state_dict = {
            "c2w": camera_state.c2w.tolist(),
            "fov": client.camera.fov,
            "aspect": camera_state.aspect,
        }
        # Ensure the image values are in the [0, 1] range and convert to 8-bit
        image = np.clip(render_rgbs, 0, 1)
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Define a threshold to detect "background" pixels.
        # Pixels with all channels below this threshold are considered background.
        threshold = 1  # adjust if needed
        background_mask = np.all(image_uint8 < threshold, axis=-1)
        
        # Replace background pixels with white (255,255,255)
        image_uint8[background_mask] = [255, 255, 255]
        
        # Save the image using Pillow
        pil_image = Image.fromarray(image_uint8)
        pil_image.save("./snapshot"+str(idx)+".png")
        tmp_dir = ".tmp"
        os.makedirs(tmp_dir, exist_ok=True)  # create directory if it doesn't exist
        camera_state_path = os.path.join(tmp_dir, "camera_state.json")
        with open(camera_state_path, "w") as f:
            json.dump(camera_state_dict, f, indent=4)
        print("Image saved to ./snapshot",idx,"+.png")
        
        idx=idx+1
        return idx
    
    def rerender_K(self, K):
        K = K.target.value / 180.0 * np.pi
        render_fn = functools.partial(self.render_fn, fov=K)
        client = self.server.get_clients()[0]
        client.camera.fov = K
        self.render_fn = render_fn
        self.rerender(None)
        
    def move(self, x):
        print(x)
        print(dir(x))
        print(x.target.value)
        self.rerender(None)

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion (w, x, y, z).
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    
    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return np.array([w, x, y, z])
