import nerfview
import viser
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
import pdb 
import trimesh
import trimesh.creation
import viser.transforms as tf
import numpy as np
from utils.color_shs import RGB2SH
from core.splat import SplatData
import functools
from models.clip_query import OpenCLIPNetwork, OpenCLIPNetworkConfig, get_text_feature, SigLIPNetwork, SigLIPNetworkConfig

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
            
        pred = torch.sigmoid(pred)
        
        print(pred, pred.min(), pred.max())
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        return pred


class LanguageFeature:
    def __init__(self, viewer, splatdata: SplatData, feature_type="siglip"):
        self.viewer = viewer
        self.server = viewer.server
        self.splats = splatdata
        self.language_feature = splatdata.get_data()['language_feature']
        self.num_class = 5
        self.prune_rate = 1.0
        self.masks = None

        self._feature_map = False
        self._normal_map = False
        self._hard_class = False
        self.gs_scores = torch.zeros(self.language_feature.shape[0])
        self.classes_colors = torch.zeros_like(self.language_feature)
        self.labels = torch.zeros(self.language_feature.shape[0])
        self.cluster_centers = None
        self.class_id = -1
        self.query_feature = None
        
        self.encoder_hidden_dims = [256, 128, 64, 32, 3]
        self.decoder_hidden_dims = [16, 32, 64, 128, 256, 256, 512]
        if feature_type == "siglip":
            self.config = SigLIPNetworkConfig()
            self.network = SigLIPNetwork(self.config)
        elif feature_type == "clip":
            self.config = OpenCLIPNetworkConfig()
            self.network = OpenCLIPNetwork(self.config)
        else:
            self.network = None
            self.query_feature = torch.tensor(np.load(feature_type)[0]).to(splatdata.device)
        # self.config = SigLIPNetworkConfig()
        # self.network = SigLIPNetwork(self.config)
        # self.config = OpenCLIPNetworkConfig()
        # self.network = OpenCLIPNetwork(self.config)
        self.language_feature_large = splatdata.get_large()
    
        with self.server.gui.add_folder(label="Language Feature"):
            self._feature_vis_button = self.server.gui.add_button("Feature Map")
            self._feature_vis_button.on_click(self._toggle_feature_map)
            
            self._text_prompt = self.server.gui.add_text("Text Prompt", initial_value="chair")
            self._text_prompt.on_update(self.update_language_feature)
            self._text_prompt_rate = self.server.gui.add_text("Rate", initial_value="0.6")
            self._text_prompt_rate.on_update(self.update_prune_rate)

            self._prune_based_on_text = self.server.gui.add_button("Prune based on text prompt")
            self._prune_based_on_text.on_click(self.prune_based_on_text)
            
            self._feature_vis_button = self.server.gui.add_button("Normal Map")
            self._feature_vis_button.on_click(self._toggle_normal_map)
            
            # self._hard_class_button = self.server.gui.add_button("Hard Label Class")
            # self._hard_class_button.on_click(self._get_hard_class)

            # self._class_number = self.server.gui.add_text("Class numbers", initial_value="5")
            # self._class_number.on_update(self.update_class_number)

            # self._check_classes_id = self.server.gui.add_text("Target Class ID", initial_value="-1")
            # self._check_classes_id.on_update(self.show_target_class)

            # self._prune_based_on_classes_id = self.server.gui.add_button("Prune based on ClassID")
            # self._prune_based_on_classes_id.on_click(self.prune_based_on_class_ids)

    def _toggle_feature_map(self, _):
        self._feature_map = True
        self.update_splat_renderer()
    
    def _toggle_normal_map(self, _):
        self._normal_map = True
        self.update_splat_renderer()

    def _get_text_feature(self, text):
        text_feature = self.network.encode_text(text)
        # text_feature = get_text_feature(self.network, text)
        # text_feature = torch.sigmoid(text_feature)
        return text_feature
        
    def update_class_number(self, num):
        self.num_clusters = int(num.target.value)
        self.update_splat_renderer()
    

    # def _get_hard_class(self, _):
    #     self._hard_class = True
    #     self._feature_map = False
    #     language_feature = self.language_feature.detach().cpu().numpy()
    #     num_clusters = self.num_class
        
    #     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    #     kmeans.fit(language_feature)
    #     self.labels = kmeans.labels_
    #     self.cluster_centers = kmeans.cluster_centers_

    #     self.classes_colors = torch.tensor(self.cluster_centers[self.labels]).to(self.language_feature.device)
    #     self.update_splat_renderer()
    
    def update_language_feature(self, text):
        text = text.target.value
        print(text)
        if text == "all":
            self.gs_scores = torch.zeros(self.language_feature.shape[0]).to(self.language_feature.device)
        elif self.query_feature is not None:
            new_feature = self.query_feature.to(torch.float)
            # new_feature = self._get_text_feature(text).to(torch.float)
            # gs_featuers = self.pca.inverse_transform(self.language_feature.detach().cpu().numpy())
            Cos = CosineClassifier()
            scores = Cos(new_feature, self.language_feature_large, scale=False).squeeze(0)
            self.gs_scores = scores
        else: 
            new_feature = self._get_text_feature(text).to(torch.float)
            # gs_featuers = self.pca.inverse_transform(self.language_feature.detach().cpu().numpy())

            Cos = CosineClassifier()
            scores = Cos(new_feature, self.language_feature_large, scale=False).squeeze(0)
            self.gs_scores = scores
        self.update_splat_renderer()
        
    def prune_based_on_text(self, _):
        if self.gs_scores.sum() == 0:
            self.masks = None
        else:
            self.masks = (self.gs_scores <= self.prune_rate)
        self.update_splat_renderer()
        
    # def show_target_class(self, text):
    #     self.class_id = int(text.target.value)

    # def prune_based_on_class_ids(self, _):
    #     if self.class_id == -1:
    #         self.masks = None
    #     else:
    #         self.masks = ~(self.class_id == self.labels)
    #     self.update_splat_renderer()
        
    def update_prune_rate(self, text):
        self.prune_rate = float(text.target.value)
        self.update_splat_renderer()

    def update_splat_renderer(self, device='cuda', backend='gsplat'):
        means, norms, quats, scales, opacities, colors, sh_degree, language_feature = self.splats.get_data().values()
        # print(self._feature_map, self._normal_map)
        if self.gs_scores.sum() != 0:
            new_colors = colors.clone()
            new_colors[self.gs_scores > self.prune_rate] = RGB2SH(torch.tensor([1, 0, 0]).to(device)).unsqueeze(0)
            # language_feature[self.gs_scores > self.prune_rate] = RGB2SH(torch.tensor([1, 0, 0]).to(device)).unsqueeze(0)
            print("there are {} points with score > {}".format((self.gs_scores > self.prune_rate).sum(), self.prune_rate))
        
        if self.masks is not None:
            mask = self.masks
            means = means[mask]
            quats = quats[mask]
            scales = scales[mask]
            opacities = opacities[mask]
            colors = colors[mask]
            # new_colors = new_colors[mask]
            classes_colors = self.classes_colors[mask]
            language_feature = self.language_feature[mask]
        
        if self._feature_map:
            render_fn = functools.partial(self.viewer.render_fn, 
                                          means=means, 
                                          quats=quats, 
                                          scales=scales,
                                          opacities=opacities,
                                          colors=language_feature,
                                          sh_degree=None,
                                          device=device,
                                          backend=backend)
        
        elif self._normal_map:
            render_fn = functools.partial(self.viewer.render_fn, 
                                          means=means, 
                                          quats=quats, 
                                          scales=scales,
                                          opacities=opacities,
                                          colors=norms,
                                          sh_degree=None,
                                          device=device,
                                          backend=backend)
        # elif self._hard_class:
        #     render_fn = functools.partial(self.viewer.render_fn, 
        #                                   means=means, 
        #                                   quats=quats, 
        #                                   scales=scales,
        #                                   opacities=opacities,
        #                                   colors=classes_colors,
        #                                   sh_degree=None,
        #                                   device=device,
        #                                   backend=backend)
        else:
            render_fn = functools.partial(self.viewer.render_fn, 
                                          means=means, 
                                          quats=quats, 
                                          scales=scales,
                                          opacities=opacities,
                                          colors=new_colors if self.gs_scores.sum() != 0 and self.masks is None else colors,
                                          sh_degree=sh_degree,
                                          device=device,
                                          backend=backend)

        self.viewer.render_fn = render_fn
        self.viewer.rerender(None) 