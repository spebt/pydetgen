import torch
import math
from torch import Tensor, tensor, pi, cos, sin, stack, asin, tan, deg2rad
from matplotlib.collections import PolyCollection
from matplotlib.axes import Axes

def plot_polygons_from_vertices_2d_mpl(vertices: Tensor, ax: Axes, **kwargs):
    """Plots a collection of polygons on a matplotlib Axes object."""
    p = PolyCollection(vertices.tolist(), **kwargs)
    ax.add_collection(p)
    return p

def rotate_and_repeat_4gon(input: Tensor, n: int, step: float) -> Tensor:
    """Rotates and repeats the vertices of a quadrilateral to form a ring."""
    m_polygons = input.shape[0]
    rotations = torch.arange(0, n, dtype=input.dtype, device=input.device) * step
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

def generate_sc_spect_detectors(cfg: dict) -> Tensor:
    """Generates the 4 concentric rings of detectors based on the SC-SPECT paper."""
    all_detectors = []
    det_w = cfg["detector_width_mm"]
    det_t = cfg["detector_thickness_mm"]
    
    for r_in, n_crystals in zip(cfg["det_rings_r_in"], cfg["det_rings_n_crystals"]):
        r_out = r_in + det_t
        r_mid = r_in + (det_t / 2.0)
        
        # Angular width of one physical crystal
        angular_width = det_w / r_mid
        angle_step = 2 * math.pi / n_crystals
        
        # Base crystal centered at angle 0
        th_start = -angular_width / 2.0
        th_end = angular_width / 2.0
        
        # Use math.cos and math.sin since th_start and th_end are standard Python floats
        base_crystal = tensor([[
            [r_in * math.cos(th_start), r_in * math.sin(th_start)],
            [r_out * math.cos(th_start), r_out * math.sin(th_start)],
            [r_out * math.cos(th_end),   r_out * math.sin(th_end)],
            [r_in * math.cos(th_end),   r_in * math.sin(th_end)],
        ]], dtype=torch.float32)
        
        # Rotate and repeat to form the full layer
        ring_crystals = rotate_and_repeat_4gon(base_crystal, n_crystals, angle_step)
        all_detectors.append(ring_crystals)
        
    return torch.cat(all_detectors, dim=0)

def generate_biconical_collimators(
    pinhole_diameter_mm: float,
    pinhole_opening_angle_deg: float,
    n_pinholes: int,
    collimator_ring_radius_mm: float,
    collimator_thickness_mm: float,
) -> tuple[Tensor, Tensor]:
    """
    Generates a perfectly focused bi-conical (double-tapered) pinhole 
    collimator across inner and outer rings.
    """
    single_ring_thickness = collimator_thickness_mm / 2.0
    center_junction_radius = collimator_ring_radius_mm

    R_inner_coll_inner = center_junction_radius - single_ring_thickness
    R_inner_coll_outer = center_junction_radius
    R_outer_coll_inner = center_junction_radius
    R_outer_coll_outer = center_junction_radius + single_ring_thickness

    # Common Geometric Calculations
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

    all_inner_segments = []
    all_outer_segments = []

    for i in range(n_pinholes):
        center_angle = i * angle_step_per_pinhole

        # 1. Inner Collimator Segment for this pinhole
        angle_seg_start_inner_ring = center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_inner_wide
        angle_seg_end_inner_ring = center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_inner_wide
        angle_seg_start_outer_ring = center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_inner_narrow
        angle_seg_end_outer_ring = center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_inner_narrow
        
        inner_verts = tensor([[
            [R_inner_coll_inner * cos(angle_seg_start_inner_ring), R_inner_coll_inner * sin(angle_seg_start_inner_ring)],
            [R_inner_coll_outer * cos(angle_seg_start_outer_ring), R_inner_coll_outer * sin(angle_seg_start_outer_ring)],
            [R_inner_coll_outer * cos(angle_seg_end_outer_ring),   R_inner_coll_outer * sin(angle_seg_end_outer_ring)],
            [R_inner_coll_inner * cos(angle_seg_end_inner_ring),   R_inner_coll_inner * sin(angle_seg_end_inner_ring)],
        ]], dtype=torch.float32)
        all_inner_segments.append(inner_verts)

        # 2. Outer Collimator Segment for this pinhole
        angle_seg_start_inner_ring_o = center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_outer_narrow
        angle_seg_end_inner_ring_o = center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_outer_narrow
        angle_seg_start_outer_ring_o = center_angle - angle_step_per_pinhole / 2.0 + angular_half_width_outer_wide
        angle_seg_end_outer_ring_o = center_angle + angle_step_per_pinhole / 2.0 - angular_half_width_outer_wide
        
        outer_verts = tensor([[
            [R_outer_coll_inner * cos(angle_seg_start_inner_ring_o), R_outer_coll_inner * sin(angle_seg_start_inner_ring_o)],
            [R_outer_coll_outer * cos(angle_seg_start_outer_ring_o), R_outer_coll_outer * sin(angle_seg_start_outer_ring_o)],
            [R_outer_coll_outer * cos(angle_seg_end_outer_ring_o),   R_outer_coll_outer * sin(angle_seg_end_outer_ring_o)],
            [R_outer_coll_inner * cos(angle_seg_end_inner_ring_o),   R_outer_coll_inner * sin(angle_seg_end_inner_ring_o)],
        ]], dtype=torch.float32)
        all_outer_segments.append(outer_verts)

    return torch.cat(all_inner_segments, dim=0), torch.cat(all_outer_segments, dim=0)