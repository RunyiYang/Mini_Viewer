import torch
import torch.nn as nn
import torch.nn.functional as F
import nerfview
import functools
from sklearn.cluster import KMeans
from plyfile import PlyData, PlyElement  # PlyFile import
import numpy as np
import pdb
import os
import viser
from core.splat import SplatData

class ViewerEditor(nerfview.Viewer):

    def __init__(self, 
                 splat_args, 
                 splat_data, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._editor = None
        self.splat_args = splat_args
        self.device = splat_args.device
        self.language_feature = splat_args.language_feature
        self.mode = "rgb"
        self.splats = splat_data
        self.adjust_viewer()
        
    def adjust_viewer(self):
        """
        Adjust the inherited viewer.
        """
        with self._rendering_folder:
            self._max_img_res_slider.remove()
            self._max_img_res_slider = self.server.gui.add_slider(
                "Max Img Res", min=64, max=2048, step=1, initial_value=512
            )
            self._max_img_res_slider.on_update(self.rerender)
            
        @self.server.on_client_connect
        def handle_client_connect(client: viser.ClientHandle):
            print("Client connected")
            # if client.client_id not in self.processed_clients:
            self.splat_central_point = self.find_splat_central_point()
            # client.camera.position = np.array([0.0, 0.0, -10.0])
            client.camera.position = self.splat_central_point + np.array([0.0, 0.0, -10.0])
            client.camera.wxyz = np.array([0.0, 0.0, 0.0, 1.0])

        


    def find_splat_central_point(self) -> np.ndarray:
        """
            Find the central point of the splat data.
        """
        means = self.splats._means
        return torch.mean(means, 0).cpu().numpy()
            # with self.server.gui.add_folder("Basic"):
            #     self._rgb = self.server.gui.add_button("RGB")
            #     self._rgb.on_click(self.get_rgb)
                
            #     self._depth = self.server.gui.add_button("Depth")
            #     self._depth.on_click(self.get_depth)
                
            #     self._alpha = self.server.gui.add_button("Normal")
            #     self._alpha.on_click(self.get_alpha)
            
            # if self.splat_args.language_feature:
            #     with self.server.gui.add_folder("Feature"):
            #         self._feature_vis_button = self.server.gui.add_button("Feature Map")
            #         self._feature_vis_button.on_click(self._toggle_feature_map)
                    
            #         # Add button to export PLY file
            #         self._export_ply_button = self.server.gui.add_button("Export PLY")
            #         self._export_ply_button.on_click(lambda _: self.save_as_ply())
                    
            # self.update_splat_renderer()

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
                                                        norms= splat_data["norms"],
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