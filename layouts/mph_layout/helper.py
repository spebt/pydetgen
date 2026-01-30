# helper.py (Corrected)
import torch
from matplotlib.collections import PolyCollection
from matplotlib.axes import Axes
from torch import Tensor, tensor, pi, cos, sin, arange, stack, asin, tan, deg2rad
from math import cos as math_cos, sin as math_sin

def plot_polygons_from_vertices_2d_mpl(vertices: Tensor, ax: Axes, **kwargs):
    """Plots a collection of polygons on a matplotlib Axes object."""
    p = PolyCollection(vertices.tolist(), **kwargs)
    ax.add_collection(p)
    return p

def rotate_and_repeat_4gon(input: Tensor, n: int, step: float) -> Tensor:
    """Rotates and repeats the vertices of quadrilaterals to form a full circle."""
    m_polygons = input.shape[0]
    rotations = arange(0, n, dtype=input.dtype, device=input.device) * step
    rotation_matrices = stack(
        (cos(rotations), -sin(rotations), sin(rotations), cos(rotations)),
        dim=1
    ).reshape(-1, 2, 2)

    tiled_input = input.repeat(n, 1, 1)
    reshaped_vertices = tiled_input.view(-1, 2).unsqueeze(-1)
    tiled_rotation_matrices = rotation_matrices.unsqueeze(1).repeat(
        1, m_polygons * 4, 1, 1
    ).view(-1, 2, 2)

    rotated_vertices = torch.bmm(
        tiled_rotation_matrices, reshaped_vertices
    ).view(-1, 4, 2)
    return rotated_vertices

def generate_transaxial_spect_geometry(
    pinhole_diameter_mm: float,
    pinhole_opening_angle_deg: float,
    n_pinholes: int,
    collimator_ring_radius_mm: float,
    collimator_thickness_mm: float,
    detector_ring_radius_mm: float,
    detector_thickness_mm: float,
    detector_crystal_width_mm: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Generates a 2D SPECT geometry with a correctly focused bi-conical 
    (double-tapered) pinhole across two concentric collimator rings.

    Returns:
        tuple[Tensor, Tensor, Tensor]:
            - detector_units
            - inner_collimator_segments
            - outer_collimator_segments
    """
    # --- Define Radii for the Two Collimator Rings ---
    single_ring_thickness = collimator_thickness_mm / 2.0
    center_junction_radius = collimator_ring_radius_mm

    R_inner_coll_inner = center_junction_radius - single_ring_thickness
    R_inner_coll_outer = center_junction_radius

    R_outer_coll_inner = center_junction_radius
    R_outer_coll_outer = center_junction_radius + single_ring_thickness

    # --- Common Geometric Calculations ---
    angle_step_per_pinhole = 2 * pi / n_pinholes
    opening_angle_rad = deg2rad(tensor(pinhole_opening_angle_deg, dtype=torch.float32))
    half_aperture_at_junction = pinhole_diameter_mm / 2.0
    
    taper_offset = single_ring_thickness * tan(opening_angle_rad / 2.0).item()
    half_width_at_wide_side = half_aperture_at_junction + taper_offset

    # Angular half-width of the pinhole void at each collimator surface
    angular_half_width_inner_wide = asin(tensor(half_width_at_wide_side / R_inner_coll_inner))
    angular_half_width_inner_narrow = asin(tensor(half_aperture_at_junction / R_inner_coll_outer))
    angular_half_width_outer_narrow = asin(tensor(half_aperture_at_junction / R_outer_coll_inner))
    angular_half_width_outer_wide = asin(tensor(half_width_at_wide_side / R_outer_coll_outer))

    # --- Generate Collimator Segments inside a Loop for Correct Focusing ---
    all_inner_segments = []
    all_outer_segments = []

    for i in range(n_pinholes):
        # Calculate the center angle for the CURRENT pinhole
        center_angle = i * angle_step_per_pinhole

        # --- 1. Generate Inner Collimator Segment for this pinhole ---
        # Calculate the absolute angles for the solid segment edges
        angle_seg_start_inner_ring = center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_inner_wide
        angle_seg_end_inner_ring = center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_inner_wide
        angle_seg_start_outer_ring = center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_inner_narrow
        angle_seg_end_outer_ring = center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_inner_narrow
        
        inner_verts = tensor([[
            [R_inner_coll_inner * cos(angle_seg_start_inner_ring), R_inner_coll_inner * sin(angle_seg_start_inner_ring)],
            [R_inner_coll_outer * cos(angle_seg_start_outer_ring), R_inner_coll_outer * sin(angle_seg_start_outer_ring)],
            [R_inner_coll_outer * cos(angle_seg_end_outer_ring),   R_inner_coll_outer * sin(angle_seg_end_outer_ring)],
            [R_inner_coll_inner * cos(angle_seg_end_inner_ring),   R_inner_coll_inner * sin(angle_seg_end_inner_ring)],
        ]])
        all_inner_segments.append(inner_verts)

        # --- 2. Generate Outer Collimator Segment for this pinhole ---
        # Calculate the absolute angles for the solid segment edges
        angle_seg_start_inner_ring_o = center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_outer_narrow
        angle_seg_end_inner_ring_o = center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_outer_narrow
        angle_seg_start_outer_ring_o = center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_outer_wide
        angle_seg_end_outer_ring_o = center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_outer_wide
        
        outer_verts = tensor([[
            [R_outer_coll_inner * cos(angle_seg_start_inner_ring_o), R_outer_coll_inner * sin(angle_seg_start_inner_ring_o)],
            [R_outer_coll_outer * cos(angle_seg_start_outer_ring_o), R_outer_coll_outer * sin(angle_seg_start_outer_ring_o)],
            [R_outer_coll_outer * cos(angle_seg_end_outer_ring_o),   R_outer_coll_outer * sin(angle_seg_end_outer_ring_o)],
            [R_outer_coll_inner * cos(angle_seg_end_inner_ring_o),   R_outer_coll_inner * sin(angle_seg_end_inner_ring_o)],
        ]])
        all_outer_segments.append(outer_verts)

    # Combine the lists of tensors into final output tensors
    inner_collimator_segments = torch.cat(all_inner_segments, dim=0)
    outer_colimator_segments = torch.cat(all_outer_segments, dim=0)

    # --- 3. Detector Geometry (This part was correct and remains unchanged) ---
    R_DET_INNER = detector_ring_radius_mm
    R_DET_OUTER = detector_ring_radius_mm + detector_thickness_mm

    # 1. Use the mid-radius for the most accurate circumference and width calculations.
    mid_radius_det = R_DET_INNER + detector_thickness_mm / 2.0
    circumference_det = 2 * pi * mid_radius_det

    # 2. Determine the number of crystals that fit without overlapping, keeping their original width.
    n_total_crystals = int(circumference_det / detector_crystal_width_mm)

    # 3. Calculate the fixed angular width of a single crystal based on its physical size.
    #    This is the angle subtended by one crystal.
    angular_width_crystal = tensor(detector_crystal_width_mm / mid_radius_det)

    # 4. Calculate the total angular space occupied by all crystals.
    total_crystal_angular_span = n_total_crystals * angular_width_crystal

    # 5. Calculate the leftover angular space (the total gap) and divide it evenly.
    total_angular_gap = tensor(2 * pi) - total_crystal_angular_span
    angular_gap_between_crystals = total_angular_gap / n_total_crystals

    # 6. The new step size is the width of one crystal PLUS one evenly-sized gap.
    angle_step_per_crystal = angular_width_crystal + angular_gap_between_crystals

    print(f"Placing {n_total_crystals} crystals with an evenly distributed angular gap of "
          f"{torch.rad2deg(tensor(angular_gap_between_crystals)):.4f} degrees between each.")

    # 7. Define the first crystal based on its INTRINSIC angular width.
    #    Note: The angular width of the polygon is now different from the step size.
    angle_start_crystal = -angular_width_crystal / 2.0
    angle_end_crystal = angular_width_crystal / 2.0

    # --- MODIFICATION END ---

    crystal_vertices = tensor([[
        [R_DET_INNER * cos(angle_start_crystal), R_DET_INNER * sin(angle_start_crystal)],
        [R_DET_OUTER * cos(angle_start_crystal), R_DET_OUTER * sin(angle_start_crystal)],
        [R_DET_OUTER * cos(angle_end_crystal),   R_DET_OUTER * sin(angle_end_crystal)],
        [R_DET_INNER * cos(angle_end_crystal),   R_DET_INNER * sin(angle_end_crystal)],
    ]])
    
    # This function correctly rotates and places the crystals using the new step size, creating the gaps.
    detector_units = rotate_and_repeat_4gon(crystal_vertices, n_total_crystals, angle_step_per_crystal)

    return detector_units, inner_collimator_segments, outer_colimator_segments



def translate_vertices_2d(vertices: Tensor, dx: float, dy: float) -> Tensor:
    """
    Translate a batch of polygons (N_polys, 4, 2) by (dx, dy) in mm.
    No inplace modification; returns a new tensor.
    """
    shift = torch.tensor([dx, dy], dtype=vertices.dtype, device=vertices.device)
    return vertices + shift

def make_transaxial_positions(n_pos: int,
                              a_mm: float,
                              b_mm: float,
                              outward_shift_mm: float = 0.0) -> Tensor:
    """
    Evenly spaced points on an ellipse of semi-axes (a_mm, b_mm) with optional uniform
    outward radial shift to enlarge the convex hull (per Chen et al.):
      16->0 mm, 8->2 mm, 6->5 mm, 4->15 mm.
    Returns: (n_pos, 2) tensor of (x_mm, y_mm).
    """
    thetas = torch.linspace(0, 2 * pi, steps=n_pos + 1, dtype=torch.float32)[:-1]
    xs = a_mm * torch.cos(thetas)
    ys = b_mm * torch.sin(thetas)
    pts = torch.stack([xs, ys], dim=1)  # (n_pos, 2)

    if outward_shift_mm != 0.0:
        radii = torch.linalg.norm(pts, dim=1, keepdims=True).clamp(min=1e-6)
        dirs = pts / radii
        pts = pts + outward_shift_mm * dirs

    return pts