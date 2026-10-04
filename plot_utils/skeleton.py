from typing import Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go

# =============================================================================
# Anatomical Upper-Body Skeleton Definition (NO Legs / Bottom)
# =============================================================================
UPPER_BODY_BONES = [
    # Spine & Torso
    ("head", "chest"),
    ("chest", "pelvis"),
    # Right Arm
    ("chest", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Left Arm
    ("chest", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
]

UPPER_BODY_JOINTS = [
    "head", "chest", "pelvis",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
]


def add_upper_body_skeleton(
    fig: go.Figure,
    kpts_dict: Dict[str, np.ndarray],
    bone_color: str = "#475569",
    joint_color: str = "#334155",
    width: int = 6,
    marker_size: int = 5,
    name: str = "Upper Body",
    opacity: float = 0.9,
    showlegend: bool = True,
    row: Optional[int] = None,
    col: Optional[int] = None,
):
    """Draws upper-body skeleton (torso, head, arms) in Plotly 3D without bottom/legs."""
    bx, by, bz = [], [], []
    for p1, p2 in UPPER_BODY_BONES:
        if p1 in kpts_dict and p2 in kpts_dict:
            v1 = np.array(kpts_dict[p1])
            v2 = np.array(kpts_dict[p2])
            bx.extend([v1[0], v2[0], None])
            by.extend([v1[1], v2[1], None])
            bz.extend([v1[2], v2[2], None])

    # Bone lines
    trace_bone = go.Scatter3d(
        x=bx, y=by, z=bz,
        mode="lines",
        line=dict(color=bone_color, width=width),
        opacity=opacity,
        name=name,
        showlegend=showlegend,
    )

    # Joint spheres
    jx = [kpts_dict[k][0] for k in UPPER_BODY_JOINTS if k in kpts_dict]
    jy = [kpts_dict[k][1] for k in UPPER_BODY_JOINTS if k in kpts_dict]
    jz = [kpts_dict[k][2] for k in UPPER_BODY_JOINTS if k in kpts_dict]

    trace_joints = go.Scatter3d(
        x=jx, y=jy, z=jz,
        mode="markers",
        marker=dict(size=marker_size, color=joint_color, opacity=opacity),
        hoverinfo="skip",
        showlegend=False,
    )

    if row and col:
        fig.add_trace(trace_bone, row=row, col=col)
        fig.add_trace(trace_joints, row=row, col=col)
    else:
        fig.add_trace(trace_bone)
        fig.add_trace(trace_joints)


def add_bone_strain_skeleton(
    fig: go.Figure,
    kpts_dict: Dict[str, np.ndarray],
    nom_bone_lens: Dict[Tuple[str, str], float],
    row: Optional[int] = None,
    col: Optional[int] = None,
    width: int = 6,
    marker_size: int = 5,
    visible: bool = True,
):
    """Draws upper-body skeleton with bones color-coded according to length strain relative to nominal."""
    for p1, p2 in UPPER_BODY_BONES:
        if p1 in kpts_dict and p2 in kpts_dict:
            v1 = np.array(kpts_dict[p1])
            v2 = np.array(kpts_dict[p2])
            l_act = float(np.linalg.norm(v1 - v2))
            l_nom = float(nom_bone_lens.get((p1, p2), 0.30))
            strain_pct = abs(l_act - l_nom) / max(l_nom, 1e-4) * 100.0

            if strain_pct <= 2.0:
                b_color = "#10B981"  # Emerald green (Rigid)
            elif strain_pct <= 10.0:
                b_color = "#F59E0B"  # Amber (Mild deformation)
            else:
                b_color = "#EF4444"  # Crimson (Severe disarticulation)

            bone_trace = go.Scatter3d(
                x=[v1[0], v2[0]], y=[v1[1], v2[1]], z=[v1[2], v2[2]],
                mode="lines",
                line=dict(color=b_color, width=width),
                hoverinfo="text",
                hovertext=f"{p1} - {p2}<br>Strain: {strain_pct:.1f}%<br>L: {l_act*100:.1f} cm (Nom: {l_nom*100:.1f} cm)",
                showlegend=False,
                visible=visible,
            )
            if row and col:
                fig.add_trace(bone_trace, row=row, col=col)
            else:
                fig.add_trace(bone_trace)

    jx = [kpts_dict[k][0] for k in UPPER_BODY_JOINTS if k in kpts_dict]
    jy = [kpts_dict[k][1] for k in UPPER_BODY_JOINTS if k in kpts_dict]
    jz = [kpts_dict[k][2] for k in UPPER_BODY_JOINTS if k in kpts_dict]
    joint_trace = go.Scatter3d(
        x=jx, y=jy, z=jz,
        mode="markers",
        marker=dict(size=marker_size, color="#334155"),
        hoverinfo="skip",
        showlegend=False,
        visible=visible,
    )
    if row and col:
        fig.add_trace(joint_trace, row=row, col=col)
    else:
        fig.add_trace(joint_trace)
