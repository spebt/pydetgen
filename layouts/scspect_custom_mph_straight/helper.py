import torch
import math
from torch import Tensor, tensor, stack
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
        (torch.cos(rotations), -torch.sin(rotations), torch.sin(rotations), torch.cos(rotations)),
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

def generate_straight_pinhole_collimators(
    pinhole_diameter_mm: float,
    n_pinholes: int,
    collimator_ring_radius_mm: float,
    collimator_thickness_mm: float,
) -> Tensor:
    """
    Generates a single solid ring with straight-bore pinholes (parallel walls).
    """
    half_aperture = pinhole_diameter_mm / 2.0
    
    # Calculate absolute inner and outer radii
    r_in = collimator_ring_radius_mm - (collimator_thickness_mm / 2.0)
    r_out = collimator_ring_radius_mm + (collimator_thickness_mm / 2.0)

    # To maintain parallel walls for the gap, the angular width of the gap 
    # must be calculated separately for the inner and outer radii.
    th_in = math.asin(half_aperture / r_in)
    th_out = math.asin(half_aperture / r_out)

    angle_step = 2 * math.pi / n_pinholes
    solid_segments = []

    for i in range(n_pinholes):
        angle_curr = i * angle_step
        angle_next = (i + 1) * angle_step

        # The solid block spans from the end of the current pinhole gap
        # to the beginning of the next pinhole gap.
        start_in = angle_curr + th_in
        end_in = angle_next - th_in
        
        start_out = angle_curr + th_out
        end_out = angle_next - th_out

        verts = tensor([[
            [r_in * math.cos(start_in), r_in * math.sin(start_in)],
            [r_out * math.cos(start_out), r_out * math.sin(start_out)],
            [r_out * math.cos(end_out),   r_out * math.sin(end_out)],
            [r_in * math.cos(end_in),   r_in * math.sin(end_in)],
        ]], dtype=torch.float32)
        
        solid_segments.append(verts)

    # Returns a single tensor containing all the solid chunks of the collimator ring
    return torch.cat(solid_segments, dim=0)