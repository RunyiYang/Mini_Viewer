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
from viewer import ViewerEditor
import functools

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, default=None, help="Instead of ckpt, provide ply from Inria and get the view")
    parser.add_argument("--port", type=int, default=8080, help="port for the viewer server")
    parser.add_argument(
        "--backend", type=str, default="gsplat", help="gsplat, gsplat_legacy, inria"
    )
    parser.add_argument("--language_feature", type=str, help="Whether to load language feature")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    torch.manual_seed(42)
    # device = "cuda"


    # register and open viewer
    splat_data = load_data(args)
    if args.language_feature:
        means, quats, scales, opacities, colors, sh_degree, language_feature = splat_data
    else:
        means, quats, scales, opacities, colors, sh_degree = splat_data

    viewer_render_fn_partial = functools.partial(viewer_render_fn, 
                                                 means=means, 
                                                 quats=quats, 
                                                 scales=scales, 
                                                 opacities=opacities, 
                                                 colors=colors, 
                                                 sh_degree=sh_degree, 
                                                 device=args.device, 
                                                 backend=args.backend,
                                                 mode="rgb",
                                                 )

    server = viser.ViserServer(port=args.port, verbose=False)

    _ = ViewerEditor(
        server=server,
        splat_args=args,
        splat_data=splat_data,
        render_fn=viewer_render_fn_partial,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    main()