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

class CosineClassifier(nn.Module):
    def __init__(self, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.temp = temp

    def forward(self, img, concept, scale=True):
        """
        img: (bs, emb_dim)
        concept: (n_class, emb_dim)
        """
        img_norm = F.normalize(img, dim=-1)
        concept_norm = F.normalize(concept, dim=-1)
        pred = torch.matmul(img_norm, concept_norm.transpose(0, 1))
        if scale:
            pred = pred / self.temp
        return pred

class ViewerEditor(nerfview.Viewer):

    def __init__(self, splat_args, splat_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._editor = None
        self._feature_map = False
        self._hard_class = False
        self.splat_args = splat_args
        self.mode = "rgb"
        self.masks = None
        if self.splat_args.language_feature:
            self.language_feature = splat_data[-1]
            self.splat_data = splat_data[:-1]
            self.gs_scores = torch.zeros(self.language_feature.shape[0])
            self.classes_colors = torch.zeros_like(self.language_feature)
            self.labels = torch.zeros(self.language_feature.shape[0])
            self.cluster_centers = None
            self.num_clusters = 5
        else:
            self.splat_data = splat_data
        
        self.adjust_viewer()

    def _toggle_feature_map(self, _):
        self._feature_map = True
        self.update_splat_renderer()

    def get_rgb(self, _):
        self._feature_map = False
        self._hard_class = False
        self.mode = "rgb"
        self.update_splat_renderer()

    def get_depth(self, _):
        self._feature_map = False
        self._hard_class = False
        self.mode = "depth"
        self.update_splat_renderer()

    def get_alpha(self, _):
        self._feature_map = False
        self._hard_class = False
        self.mode = "normal"
        self.update_splat_renderer()
        
        
    def update_class_number(self, num):
        self.num_clusters = int(num.target.value)
        self.update_splat_renderer()

    def save_as_ply(self, output_ply_filename="pruned_output.ply"):
        """
        Reads the PLY file from self.splat_args.ply, applies a mask to prune the data, 
        and saves the pruned data to a new PLY file.
        """

        def mkdir_p(path):
            """Creates the output directory if it doesn't exist."""
            os.makedirs(path, exist_ok=True)

        # Read the input PLY file from self.splat_args.ply
        input_ply_filename = self.splat_args.ply
        ply_data = PlyData.read(input_ply_filename)
        
        # Extract vertex data
        vertex_data = ply_data['vertex'].data
        num_vertices = len(vertex_data)

        # Check if the mask exists
        if self.masks is not None:
            mask = self.masks.cpu().numpy()  # Assuming the mask is a PyTorch tensor
            assert len(mask) == num_vertices, "Mask length must match the number of vertices in the PLY file."

            # Prune the vertex data by applying the mask
            pruned_vertex_data = vertex_data[mask]  # Keep only vertices where the mask is non-zero
        else:
            # If no mask is provided, use all the data
            pruned_vertex_data = vertex_data

        # Prepare the output directory
        # mkdir_p(os.path.dirname(output_ply_filename))

        # Create a PlyElement from the pruned data
        pruned_vertex_element = PlyElement.describe(pruned_vertex_data, 'vertex')

        # Write the pruned data to the new PLY file
        PlyData([pruned_vertex_element], text=False).write(output_ply_filename)

        print(f"PLY file saved as {output_ply_filename}")
        


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

            with self.server.gui.add_folder("Basic"):
                self._rgb = self.server.gui.add_button("RGB")
                self._rgb.on_click(self.get_rgb)
                
                self._depth = self.server.gui.add_button("Depth")
                self._depth.on_click(self.get_depth)
                
                self._alpha = self.server.gui.add_button("Normal")
                self._alpha.on_click(self.get_alpha)
            
            if self.splat_args.language_feature:
                with self.server.gui.add_folder("Feature"):
                    self._feature_vis_button = self.server.gui.add_button("Feature Map")
                    self._feature_vis_button.on_click(self._toggle_feature_map)
                    
                    # Add button to export PLY file
                    self._export_ply_button = self.server.gui.add_button("Export PLY")
                    self._export_ply_button.on_click(lambda _: self.save_as_ply())
                    
            self.update_splat_renderer()

    def update_splat_renderer(self, 
                              device='cuda',
                              backend='gsplat'):
        means, quats, scales, opacities, colors, sh_degree = self.splat_data

        if self.masks is None:
            if self.splat_args.language_feature:
                language_feature = self.language_feature
                classes_colors = self.classes_colors
        else:
            mask = self.masks
            means = means[mask]
            quats = quats[mask]
            scales = scales[mask]
            opacities = opacities[mask]
            colors = colors[mask]
            if self.splat_args.language_feature:
                classes_colors = self.classes_colors[mask]
                language_feature = self.language_feature[mask]

        if self._feature_map:
            render_fn = functools.partial(self.render_fn, 
                                          means=means, 
                                          quats=quats, 
                                          scales=scales,
                                          opacities=opacities,
                                          colors=language_feature,
                                          sh_degree=None,
                                          device=device,
                                          backend=backend,
                                          mode=self.mode)
        elif self._hard_class:
            render_fn = functools.partial(self.render_fn, 
                                          means=means, 
                                          quats=quats, 
                                          scales=scales,
                                          opacities=opacities,
                                          colors=classes_colors,
                                          sh_degree=None,
                                          device=device,
                                          backend=backend,
                                          mode=self.mode)
        else:
            render_fn = functools.partial(self.render_fn, 
                                          means=means, 
                                          quats=quats, 
                                          scales=scales,
                                          opacities=opacities,
                                          colors=colors,
                                          sh_degree=sh_degree,
                                          device=device,
                                          backend=backend,
                                          mode=self.mode)

        self.render_fn = render_fn
        self.rerender(None)
