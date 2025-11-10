import hashlib
import math
import re
from functools import lru_cache
from typing import Tuple

import colorsys
import viser

from .geometry import add_bounding_box


def add_script_bboxes(
    server: viser.ViserServer,
    script_text: str,
    *,
    default_wall_thickness: float = 0.15,
    default_opening_thickness: float = 0.12,
    path_prefix: str = "/scene",
) -> None:
    """Parse the custom script format and add the corresponding bounding boxes."""
    FLOAT = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'
    wall_pat = re.compile(
        rf"""(?P<name>\w+)\s*=\s*Wall\(
            \s*(?P<ax>{FLOAT})\s*,\s*(?P<ay>{FLOAT})\s*,\s*(?P<az>{FLOAT})
            \s*,\s*(?P<bx>{FLOAT})\s*,\s*(?P<by>{FLOAT})\s*,\s*(?P<bz>{FLOAT})
            \s*,\s*(?P<h>{FLOAT})
            (?:\s*,\s*(?P<th>{FLOAT}))?
            \s*\)""",
        re.VERBOSE | re.IGNORECASE,
    )

    walls = {}
    for match in wall_pat.finditer(script_text):
        ax, ay, az = map(float, (match["ax"], match["ay"], match["az"]))
        bx, by, bz = map(float, (match["bx"], match["by"], match["bz"]))
        height = float(match["h"])
        thickness = float(match["th"]) if match["th"] is not None else default_wall_thickness
        dx, dy = (bx - ax), (by - ay)
        length = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        cx, cy, cz = (ax + bx) / 2.0, (ay + by) / 2.0, (az + bz) / 2.0 + height / 2.0
        base_z = min(az, bz)
        top_z = base_z + height
        walls[match["name"]] = dict(
            center=(cx, cy, cz),
            length=length,
            height=height,
            angle_z=angle,
            thickness=thickness,
            base_z=base_z,
            top_z=top_z,
        )

    for name, wall in walls.items():
        add_bounding_box(
            server,
            position_x=wall["center"][0],
            position_y=wall["center"][1],
            position_z=wall["center"][2],
            angle_z=wall["angle_z"],
            scale_x=wall["length"],
            scale_y=wall["thickness"],
            scale_z=wall["height"],
            path=f"{path_prefix}/wall/{name}",
            color=(200, 200, 200),
            label=name,
            label_position="above",
        )

    door_pat = re.compile(
        rf"""(?P<name>\w+)\s*=\s*Door\(
            \s*(?P<wall>\w+)
            \s*,\s*(?P<px>{FLOAT})\s*,\s*(?P<py>{FLOAT})\s*,\s*(?P<pz>{FLOAT})
            \s*,\s*(?P<w>{FLOAT})\s*,\s*(?P<h>{FLOAT})
            \s*\)""",
        re.VERBOSE | re.IGNORECASE,
    )

    for match in door_pat.finditer(script_text):
        wall_info = walls.get(match["wall"], {})
        angle = wall_info.get("angle_z", 0.0)
        wall_thickness = wall_info.get("thickness", default_opening_thickness)
        px, py, pz = map(float, (match["px"], match["py"], match["pz"]))
        width, height = map(float, (match["w"], match["h"]))
        add_bounding_box(
            server,
            position_x=px,
            position_y=py,
            position_z=pz,
            angle_z=angle,
            scale_x=width,
            scale_y=min(wall_thickness, default_opening_thickness),
            scale_z=height,
            path=f"{path_prefix}/door/{match['name']}",
            color=(255, 165, 0),
            label=match["name"],
            label_position="front",
        )

    window_pat = re.compile(
        rf"""(?P<name>\w+)\s*=\s*Window\(
            \s*(?P<wall>\w+)
            \s*,\s*(?P<px>{FLOAT})\s*,\s*(?P<py>{FLOAT})\s*,\s*(?P<pz>{FLOAT})
            \s*,\s*(?P<w>{FLOAT})\s*,\s*(?P<h>{FLOAT})
            \s*\)""",
        re.VERBOSE | re.IGNORECASE,
    )

    for match in window_pat.finditer(script_text):
        wall_info = walls.get(match["wall"], {})
        angle = wall_info.get("angle_z", 0.0)
        wall_thickness = wall_info.get("thickness", default_opening_thickness)
        px, py, pz = map(float, (match["px"], match["py"], match["pz"]))
        width, height = map(float, (match["w"], match["h"]))
        center_z = pz
        add_bounding_box(
            server,
            position_x=px,
            position_y=py,
            position_z=center_z,
            angle_z=angle,
            scale_x=width,
            scale_y=min(wall_thickness, default_opening_thickness * 0.8),
            scale_z=height,
            path=f"{path_prefix}/window/{match['name']}",
            color=(0, 191, 255),
            label=match["name"],
            label_position="front",
        )

    bbox_pat = re.compile(
        rf"""(?P<name>\w+)\s*=\s*Bbox\(
            \s*(?P<class_u>[A-Za-z_]\w*)   # bare identifier only, no quotes
            \s*,\s*(?P<px>{FLOAT})\s*,\s*(?P<py>{FLOAT})\s*,\s*(?P<pz>{FLOAT})
            \s*,\s*(?P<ang>{FLOAT})\s*,\s*(?P<sx>{FLOAT})\s*,\s*(?P<sy>{FLOAT})\s*,\s*(?P<sz>{FLOAT})
            \s*\)""",
        re.VERBOSE | re.IGNORECASE,
    )

    for match in bbox_pat.finditer(script_text):
        label = match.group("class_u") or match.group("name")
        px, py, pz = map(float, (match["px"], match["py"], match["pz"]))
        ang = float(match["ang"])
        sx, sy, sz = map(float, (match["sx"], match["sy"], match["sz"]))
        add_bounding_box(
            server,
            position_x=px,
            position_y=py,
            position_z=pz,
            angle_z=ang,
            scale_x=sx,
            scale_y=sy,
            scale_z=sz,
            path=f"{path_prefix}/bbox/{match['name']}",
            color=_color_for_class(label),
            label=label,
            label_position="above",
        )


@lru_cache(maxsize=None)
def _color_for_class(label: str) -> Tuple[int, int, int]:
    """Assign a deterministic, visually distinct color for a class label."""
    digest = hashlib.md5(label.encode("utf-8")).digest()
    hue = (digest[0] / 255.0 + digest[1] / 255.0 / 256.0) % 1.0
    saturation = 0.5 + (digest[2] / 255.0) * 0.4  # keep moderately saturated
    value = 0.6 + (digest[3] / 255.0) * 0.35  # avoid very dark colors
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


__all__ = ["add_script_bboxes"]
