"""A simple example to render a (large-scale) Gaussian Splats
Found in gsplat/examples/simple_viewer.py

Originally from nerfview
```
"""

import argparse
import time
import torch
import viser
from core.renderer import viewer_render_fn
from data_loader import load_data
from core.viewer import ViewerEditor
from core.splat import SplatData
from actions.language_feature import LanguageFeature
import functools
from actions.base import BasicFeature

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, default=None, help="Instead of ckpt, provide ply from Inria and get the view")
    parser.add_argument("--port", type=int, default=1219, help="port for the viewer server")
    parser.add_argument("--language_feature", type=str, help="Whether to load language feature")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--folder_npy", type=str, default=None, help="npy folder to load the data")
    parser.add_argument("--prune", type=str, help="Whether to prune the data")
    parser.add_argument("--feature_type", type=str, default="siglip", help="clip or siglip")
    parser.add_argument(
        "--scene_list",
        type=str,
        default=None,
        help=(
            "Comma-separated list of scene ids that we want to "
            "switch between, e.g. 09c1414f1b,3db0a1c8f3,fb5a96b1a2 "
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    # device = "cuda"


    # register and open viewer
    splats = SplatData(args=args)

    viewer_render_fn_partial = functools.partial(viewer_render_fn, 
                                                 means=splats._means, 
                                                 quats=splats._quats, 
                                                 scales=splats._scales, 
                                                 opacities=splats._opacities, 
                                                 colors=splats._colors, 
                                                 sh_degree=splats._sh_degree, 
                                                 device=args.device, 
                                                 backend="gsplat",
                                                 render_mode="rgb",
                                                 )

    server = viser.ViserServer(port=args.port, verbose=False)
    server.scene.world_axes.visible = False

    # ---------------------------------------------------------------------
    # --scene_list → Python list[str]
    # ---------------------------------------------------------------------
    if args.scene_list:
        token_str = args.scene_list.strip()
        if token_str.startswith("[") and token_str.endswith("]"):
            token_str = token_str[1:-1]
        args.scene_list = [s.strip() for s in token_str.split(",") if s.strip()]
    else:
        args.scene_list = None
    
    # make sure the *current* scene is included so the drop-down is valid
    if args.scene_list and args.folder_npy:
        import os
    
        cur_scene = os.path.basename(args.folder_npy.rstrip("/"))
        if cur_scene not in args.scene_list:
            args.scene_list.insert(0, cur_scene)

    viewer_editor = ViewerEditor(
        server=server,
        splat_args=args,
        splat_data=splats,
        render_fn=viewer_render_fn_partial,
        mode="rendering",
        scene_list=args.scene_list,
    )
    
    base = BasicFeature(viewer_editor, splats)
    language_feature = LanguageFeature(viewer_editor, splats, feature_type=args.feature_type)

    viewer_editor.base_feature_panel       = base
    viewer_editor.language_feature_panel   = language_feature
    
    # server.scene.add_frame('origin')
    # server.scene.add_grid('grid', plane='xz')
    
    print("Viewer running... Ctrl+C to exit.")
    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()