from core.splat import SplatData

class BasicFeature:
    def __init__(self, viewer, splatdata:SplatData):
        self.selected_indices = None
        self.server = viewer.server
        self.viewer = viewer
        self.splatdata = splatdata

        with self.server.gui.add_folder(label="Basic"):
            self._rgb = self.server.gui.add_button('RGB')
            self._depth = self.server.gui.add_button('Depth')
            self._normal = self.server.gui.add_button('Normal')
            self._rgb.on_click(self.get_rgb)
            self._depth.on_click(self.get_depth)
            self._normal.on_click(self.get_normal)
      
    def get_rgb(self, _):
        self._feature_map = False
        self._hard_class = False
        self.mode = "rgb"
        self.viewer.update_splat_renderer(splats=self.splatdata, render_mode=self.mode)
        
    def get_depth(self, _):
        self._feature_map = False
        self._hard_class = False
        self.mode = "depth"
        self.viewer.update_splat_renderer(splats=self.splatdata, render_mode=self.mode)

    def get_normal(self, _):
        self._feature_map = False
        self._hard_class = False
        self.mode = "normal"
        self.viewer.update_splat_renderer(splats=self.splatdata, render_mode=self.mode)