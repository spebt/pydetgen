# helper_sct_3d.py
from __future__ import annotations
import hashlib
import math
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.axes import Axes


# ---------------------------------------------------
# MD5 fingerprinting (same as helper_3d.py)
# ---------------------------------------------------

def generate_md5_from_tensors(*tensors: Tensor) -> str:
    """Compute an MD5 fingerprint from the raw bytes of one or more tensors.

    Forces contiguous memory before extracting bytes so that two mathematically
    identical tensors that differ only in memory layout (e.g. one created via a
    slice/view) always produce the same hash.
    """
    h = hashlib.md5()
    for t in tensors:
        h.update(t.contiguous().detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------
# Core geometry primitive: box → hexahedron
# ---------------------------------------------------

def box_to_hex(
    cx: float,
    cy: float,
    cz: float,
    dx: float,
    dy: float,
    dz: float,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Convert a box (center + dimensions) to an 8-vertex hexahedron.

    Coordinate system for the SCT flat-panel layout:
      X: transverse (left-right)
      Y: depth (increasing away from FOV toward detector)
      Z: axial (up-down)

    Vertex ordering — front (y_min) face first, matching extrude_quads_to_hexahedra convention
    from mph_layout_3d (z_min ↔ y_min in the flat-panel frame):

      v0: (-x, y_min, -z)    v1: (+x, y_min, -z)
      v2: (+x, y_min, +z)    v3: (-x, y_min, +z)
      v4: (-x, y_max, -z)    v5: (+x, y_max, -z)
      v6: (+x, y_max, +z)    v7: (-x, y_max, +z)

    pymatcal OBB axes derived from v0, v1, v3, v4:
      e_x  = v1 - v0 = ( dx,  0,   0)   (transverse)
      e_z  = v3 - v0 = (  0,  0,  dz)   (axial)
      e_y  = v4 - v0 = (  0, dy,   0)   (depth / "radial" for flat panel)

    Back-face detection in pymatcal (max radial-XY distance): since all crystals have y > 0
    and y_max > y_min, the y_max face always has a larger sqrt(x²+y²) → correctly identified
    as back face and excluded from the 5-face solid-angle sum.
    """
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    verts = torch.tensor(
        [
            [cx - hx, cy - hy, cz - hz],  # v0
            [cx + hx, cy - hy, cz - hz],  # v1
            [cx + hx, cy - hy, cz + hz],  # v2
            [cx - hx, cy - hy, cz + hz],  # v3
            [cx - hx, cy + hy, cz - hz],  # v4
            [cx + hx, cy + hy, cz - hz],  # v5
            [cx + hx, cy + hy, cz + hz],  # v6
            [cx - hx, cy + hy, cz + hz],  # v7
        ],
        dtype=dtype,
    )
    return verts  # (8, 3)


# ---------------------------------------------------
# Detector generation
# ---------------------------------------------------

def generate_sct_detector_hexes(
    cfg: Dict,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Generate all detector crystal hexahedra for the SCT flat-panel system.

    Layers stack along the Y (depth) axis starting at cfg["detector_y_start_mm"]
    (world coordinate = collimator back face + gap).  Each crystal grid is
    centred at (0, center_y, 0) in XZ.

    Returns:
        (N_total, 8, 3) float32 tensor of crystal hexahedra.
    """
    hexes: List[Tensor] = []
    current_y = cfg["detector_y_start_mm"]
    gap_y = cfg["detector_layer_gap_mm"]

    for layer in cfg["detector_layers"]:
        nx: int = layer["nx"]
        nz: int = layer["nz"]
        dx: float = layer["crystal_x_mm"]
        dy: float = layer["crystal_y_mm"]
        dz: float = layer["crystal_z_mm"]
        pitch_x: float = layer["pitch_x_mm"]
        pitch_z: float = layer["pitch_z_mm"]
        center_y = current_y + dy / 2.0

        for ri in range(nz):
            for ci in range(nx):
                pos_x = (ci - (nx - 1) / 2.0) * pitch_x
                pos_z = (ri - (nz - 1) / 2.0) * pitch_z
                hexes.append(
                    box_to_hex(pos_x, center_y, pos_z, dx, dy, dz, dtype=dtype)
                )

        current_y += dy + gap_y

    return torch.stack(hexes, dim=0)  # (N, 8, 3)


# ---------------------------------------------------
# Collimator generation
# ---------------------------------------------------

def generate_sct_collimator_hex(
    cfg: Dict,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Represent the full collimator plate as a single hexahedron.

    NOTE: Cylindrical holes cannot be represented as hexahedra.  The plate
    bounding-box is stored in 'plate segments 3d'; the actual hole positions
    are stored separately under the 'collimator holes' key in the .tensor file.
    Any downstream pipeline that uses this geometry for attenuation modelling
    must apply the hole data separately (e.g. subtract hole transmission from
    the plate attenuation along each ray).

    Returns:
        (1, 8, 3) float32 tensor.
    """
    cy = cfg["collimator_y_center_mm"]
    plate = box_to_hex(
        0.0, cy, 0.0,
        cfg["collimator_width_x_mm"],
        cfg["collimator_thickness_y_mm"],
        cfg["collimator_height_z_mm"],
        dtype=dtype,
    )
    return plate.unsqueeze(0)  # (1, 8, 3)


def generate_sct_hole_prisms(
    holes: Tensor,
    n_sides: int = 8,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Tensor]:
    """
    Convert cylindrical hole metadata into convex prism descriptors compatible
    with pymatcal's build_convex_union_convex_poly_block / load_scanner_geometry_3d_from_layout.

    Each cylindrical hole is approximated as a regular n_sides-gon prism
    (default: octagonal, 8-sided).  The prism is axis-aligned along Y (the
    collimator depth axis) and centred at the hole centre.

    Plane convention matches pymatcal's ray_convex_polyhedron_intersection_local:
        n · x_local <= d   (x_local relative to prism centre, R = Identity)

    Planes per prism (N = n_sides + 2 total):
        Side face i  (i = 0 .. N-1):
            n_i = (cos(i·2π/N),  0,  sin(i·2π/N))
            d_i = radius   (circumscribed polygon — prism contains the cylinder)
        Top cap  (+Y):   n = (0,  1,  0),  d = half_height
        Bottom cap (−Y): n = (0, −1,  0),  d = half_height

    The returned dict also includes tight local-frame AABBs
    (poly_local_aabb_min/max) so that the broad-phase in pymatcal can
    cull hole candidates efficiently.

    Parameters
    ----------
    holes : Tensor, shape (M, 5)
        Each row: [x_center, y_front, y_back, z_center, radius_mm]
        as produced by generate_sct_collimator_holes().
    n_sides : int
        Number of polygon sides.  8 gives 98.5% area accuracy vs a circle.
    dtype : torch.dtype
        Output tensor dtype.  Use float32 to match pymatcal's DTYPE.

    Returns
    -------
    dict with keys:
        centers              : (M, 3)
        rotations            : (M, 3, 3) — identity matrices
        plane_normals_local  : (M, N_planes, 3)
        plane_offsets_local  : (M, N_planes)
        num_planes           : (M,)  int64, always N_planes
        poly_local_aabb_min  : (M, 3) — tight local AABB lower corner
        poly_local_aabb_max  : (M, 3) — tight local AABB upper corner
    """
    M = holes.shape[0]
    x_c   = holes[:, 0]
    y_f   = holes[:, 1]
    y_b   = holes[:, 2]
    z_c   = holes[:, 3]
    r     = holes[:, 4]
    cy    = (y_f + y_b) * 0.5
    half_h = (y_b - y_f) * 0.5

    # Prism centres in world space
    centers = torch.stack([x_c, cy, z_c], dim=1).to(dtype=dtype)          # (M, 3)

    # Identity rotation: prisms are axis-aligned, no transform needed
    rotations = (
        torch.eye(3, dtype=dtype).unsqueeze(0).expand(M, -1, -1).contiguous()
    )  # (M, 3, 3)

    N_planes = n_sides + 2
    plane_normals = torch.zeros(M, N_planes, 3, dtype=dtype)
    plane_offsets = torch.zeros(M, N_planes, dtype=dtype)

    # Side-face normals: (cos θᵢ, 0, sin θᵢ) for θᵢ = i · 2π/N
    angles = torch.arange(n_sides, dtype=dtype) * (2.0 * math.pi / n_sides)
    side_normals = torch.stack(
        [angles.cos(), torch.zeros(n_sides, dtype=dtype), angles.sin()], dim=1
    )  # (n_sides, 3)

    plane_normals[:, :n_sides, :] = side_normals.unsqueeze(0).expand(M, -1, -1)
    plane_offsets[:, :n_sides]    = r.unsqueeze(1).expand(-1, n_sides)

    # Cap normals (last two planes)
    plane_normals[:, n_sides,     1] =  1.0   # top cap:    +Y
    plane_normals[:, n_sides + 1, 1] = -1.0   # bottom cap: −Y
    plane_offsets[:, n_sides]         = half_h
    plane_offsets[:, n_sides + 1]     = half_h

    num_planes = torch.full((M,), N_planes, dtype=torch.int64)

    # Tight local-frame AABB: square in XZ (conservative vs octagon, fine for culling)
    poly_local_aabb_min = torch.stack([-r, -half_h, -r], dim=1).to(dtype=dtype)
    poly_local_aabb_max = torch.stack([ r,  half_h,  r], dim=1).to(dtype=dtype)

    return {
        "centers":             centers,
        "rotations":           rotations,
        "plane_normals_local": plane_normals,
        "plane_offsets_local": plane_offsets,
        "num_planes":          num_planes,
        "poly_local_aabb_min": poly_local_aabb_min,
        "poly_local_aabb_max": poly_local_aabb_max,
    }


def generate_sct_collimator_holes(
    cfg: Dict,
    seed: Optional[int] = 42,
) -> Tensor:
    """
    Generate random non-overlapping cylindrical hole positions using a
    grid-accelerated Poisson disk sampling algorithm.

    Replaces the original O(N²) brute-force rejection sampler.  A background
    grid with cell size ``min_dist / sqrt(2)`` is used so that each candidate
    only needs to be checked against the (at most) 25 cells in its 5×5
    neighbourhood — reducing per-candidate work from O(N) to O(1) and making
    the total algorithm O(K) where K is the number of candidates tried.

    This matters when the open-area fraction is high (≥ 30 %).  At low
    densities the two algorithms are equivalent in speed; at high densities
    this version is orders of magnitude faster and guaranteed to terminate
    (up to ``max_attempts``).

    Parameters
    ----------
    cfg : dict
        Must contain the keys used below.  Uses ``hole_seed`` if present,
        otherwise falls back to the ``seed`` argument.
    seed : int or None
        RNG seed for reproducible placement (default 42).

    Returns
    -------
    Tensor, shape (M, 5), float32
        Columns: [x_center, y_front, y_back, z_center, radius_mm].
        y_front / y_back are world-space Y extents (= collimator faces).
    """
    width_x: float  = cfg["collimator_width_x_mm"]
    thickness_y: float = cfg["collimator_thickness_y_mm"]
    height_z: float = cfg["collimator_height_z_mm"]
    cy: float       = cfg["collimator_y_center_mm"]
    radius: float   = cfg["hole_radius_mm"]
    num_holes: int  = cfg["num_holes"]

    y_front = cy - thickness_y / 2.0
    y_back  = cy + thickness_y / 2.0

    # Valid placement region (hole centres must stay a full radius from each edge)
    x_lo, x_hi = -(width_x  / 2.0) + radius, (width_x  / 2.0) - radius
    z_lo, z_hi = -(height_z / 2.0) + radius, (height_z / 2.0) - radius

    area = (x_hi - x_lo) * (z_hi - z_lo)
    max_packing = area / (math.pi * radius ** 2)
    if num_holes > max_packing:
        raise ValueError(
            f"Cannot fit {num_holes} non-overlapping holes of radius {radius} mm "
            f"in a {width_x}×{height_z} mm area (theoretical max ≈ {max_packing:.0f})."
        )

    # Background grid: cell_size = min_dist / sqrt(2) guarantees the 5×5
    # neighbourhood covers the full exclusion zone of radius min_dist = 2*r.
    min_dist = 2.0 * radius
    cell_size = min_dist / math.sqrt(2.0)

    grid: Dict[Tuple[int, int], Tuple[float, float]] = {}

    def _cell(px: float, pz: float) -> Tuple[int, int]:
        return (int(math.floor((px - x_lo) / cell_size)),
                int(math.floor((pz - z_lo) / cell_size)))

    def _is_valid(px: float, pz: float) -> bool:
        gx, gz = _cell(px, pz)
        min_dist_sq = min_dist * min_dist
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                nb = grid.get((gx + dx, gz + dz))
                if nb is not None:
                    if (px - nb[0]) ** 2 + (pz - nb[1]) ** 2 < min_dist_sq:
                        return False
        return True

    rng = random.Random(seed)
    placed: List[Tuple[float, float]] = []

    # Generous attempt budget: even at 40% open area this is never exhausted.
    max_attempts = max(num_holes * 2_000, 500_000)
    attempts = 0

    while len(placed) < num_holes:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Poisson disk sampling failed to place {num_holes} holes of radius "
                f"{radius} mm after {max_attempts} attempts. "
                "Reduce num_holes or hole_radius_mm (open-area fraction may be too high)."
            )
        px = rng.uniform(x_lo, x_hi)
        pz = rng.uniform(z_lo, z_hi)
        if _is_valid(px, pz):
            placed.append((px, pz))
            grid[_cell(px, pz)] = (px, pz)
        attempts += 1

    rows = [[hx, y_front, y_back, hz, radius] for hx, hz in placed]
    return torch.tensor(rows, dtype=torch.float32)  # (M, 5)


# ---------------------------------------------------
# Matplotlib 3D helpers (mirrored from helper_3d.py)
# ---------------------------------------------------

def _hex_quad_faces(hexes: Tensor) -> Tensor:
    """Extract 6 quad faces per hexahedron → (N*6, 4, 3)."""
    idx = torch.tensor(
        [
            [0, 1, 2, 3],  # y-  (front / entrance face)
            [4, 5, 6, 7],  # y+  (back face)
            [0, 1, 5, 4],  # z-
            [3, 2, 6, 7],  # z+
            [0, 4, 7, 3],  # x-
            [1, 2, 6, 5],  # x+
        ],
        dtype=torch.long,
        device=hexes.device,
    )
    return hexes[:, idx, :]  # (N, 6, 4, 3)


def plot_hexahedra_mpl3d(
    hexes: Tensor,
    ax: Axes,
    *,
    facecolor: str = "C0",
    edgecolor: str = "k",
    alpha: float = 0.35,
    wireframe: bool = False,
) -> Optional[Poly3DCollection]:
    """Render hexahedra as 6 quads each (optionally wireframe)."""
    if hexes.numel() == 0:
        return None
    faces = _hex_quad_faces(hexes).reshape(-1, 4, 3).detach().cpu().tolist()
    coll = Poly3DCollection(
        faces,
        facecolors="none" if wireframe else facecolor,
        edgecolors=edgecolor,
        linewidths=0.6 if wireframe else 0.2,
        alpha=alpha,
    )
    ax.add_collection3d(coll)
    return coll


def autoscale_3d_to_points(ax: Axes, *point_sets: Tensor, margin: float = 0.05):
    """Set equal-aspect 3D axes limits to fit all provided point tensors."""
    pts = [p.reshape(-1, 3) for p in point_sets if p is not None and p.numel() > 0]
    if not pts:
        return
    P = torch.cat(pts, dim=0).detach().cpu()
    mins = P.min(dim=0).values
    maxs = P.max(dim=0).values
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max().item() * (0.5 + margin)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
