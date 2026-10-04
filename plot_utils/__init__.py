from plot_utils.dpi import update_rcparams, configure_high_dpi_window
from plot_utils.surfaces import create_3d_tube_surface
from plot_utils.skeleton import (
    UPPER_BODY_BONES,
    UPPER_BODY_JOINTS,
    add_upper_body_skeleton,
    add_bone_strain_skeleton,
)

__all__ = [
    "update_rcparams",
    "configure_high_dpi_window",
    "create_3d_tube_surface",
    "UPPER_BODY_BONES",
    "UPPER_BODY_JOINTS",
    "add_upper_body_skeleton",
    "add_bone_strain_skeleton",
]
