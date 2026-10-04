from typing import Tuple
import numpy as np
from scipy.interpolate import splprep, splev


def create_3d_tube_surface(
    mean_trajectory: np.ndarray,
    radii: np.ndarray,
    n_theta: int = 32,
    n_fine: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a (n_fine + 2 × n_theta + 1) smooth surface grid for Plotly go.Surface.

    go.Surface uses bilinear Gouraud shading — smoothly interpolates across the entire
    surface with zero polygon-boundary artifacts, eliminating the coil/ribbed appearance
    that Mesh3d produces from flat-shaded triangles.

    Args:
        mean_trajectory: 3D centerline trajectory of shape (H, 3).
        radii: Radius profile array of shape (H,).
        n_theta: Number of angular slices around the tube perimeter.
        n_fine: Number of longitudinal points along the spline centerline.

    Returns:
        X, Y, Z: 2D coordinate mesh grids of shape (n_fine + 2, n_theta + 1) with closed end caps.
    """
    H = len(mean_trajectory)
    if H < 2:
        return np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))

    # 1. Cubic spline interpolation of centerline
    try:
        tck, _ = splprep(
            [mean_trajectory[:, 0], mean_trajectory[:, 1], mean_trajectory[:, 2]],
            s=0, k=min(3, H - 1),
        )
        u_fine = np.linspace(0, 1, n_fine)
        fine_pts = np.column_stack(splev(u_fine, tck))
    except Exception:
        u_coarse = np.linspace(0, 1, H)
        u_fine = np.linspace(0, 1, n_fine)
        fine_pts = np.column_stack(
            [np.interp(u_fine, u_coarse, mean_trajectory[:, d]) for d in range(3)]
        )

    # 2. Smooth radius profile
    u_coarse = np.linspace(0, 1, len(radii))
    r_interp = np.interp(u_fine, u_coarse, radii)
    k_size = 11
    kernel = np.ones(k_size) / k_size
    r_smooth = np.convolve(r_interp, kernel, mode="same")
    r_smooth[: k_size // 2] = r_smooth[k_size // 2]
    r_smooth[-k_size // 2 :] = r_smooth[-k_size // 2 - 1]

    # 3. Parallel-transport frame (eliminates twisting)
    tangents = np.zeros_like(fine_pts)
    tangents[0] = fine_pts[1] - fine_pts[0]
    tangents[-1] = fine_pts[-1] - fine_pts[-2]
    for i in range(1, n_fine - 1):
        tangents[i] = fine_pts[i + 1] - fine_pts[i - 1]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tangents[0], ref)) > 0.85:
        ref = np.array([0.0, 1.0, 0.0])

    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)
    for i in range(n_fine):
        t = tangents[i]
        n = np.cross(t, ref)
        if np.linalg.norm(n) < 1e-4:
            n = np.cross(t, np.array([1.0, 0.0, 0.0]))
        n /= np.linalg.norm(n) + 1e-8
        b = np.cross(t, n)
        b /= np.linalg.norm(b) + 1e-8
        normals[i] = n
        binormals[i] = b
        ref = n  # transport

    # 4. Build surface grid with closed end-caps (n_fine + 2, n_theta + 1)
    theta = np.linspace(0, 2.0 * np.pi, n_theta + 1, endpoint=True)
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    X = np.zeros((n_fine + 2, n_theta + 1))
    Y = np.zeros((n_fine + 2, n_theta + 1))
    Z = np.zeros((n_fine + 2, n_theta + 1))

    # Start cap: center point at fine_pts[0]
    X[0, :] = fine_pts[0, 0]
    Y[0, :] = fine_pts[0, 1]
    Z[0, :] = fine_pts[0, 2]

    for i in range(n_fine):
        r = max(float(r_smooth[i]), 1e-4)
        offset = r * (np.outer(cos_th, normals[i]) + np.outer(sin_th, binormals[i]))
        X[i + 1] = fine_pts[i, 0] + offset[:, 0]
        Y[i + 1] = fine_pts[i, 1] + offset[:, 1]
        Z[i + 1] = fine_pts[i, 2] + offset[:, 2]

    # End cap: center point at fine_pts[-1]
    X[-1, :] = fine_pts[-1, 0]
    Y[-1, :] = fine_pts[-1, 1]
    Z[-1, :] = fine_pts[-1, 2]

    return X, Y, Z
