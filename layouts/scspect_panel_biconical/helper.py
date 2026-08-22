"""Geometry helpers for the 2-D panel-based SC-SPECT configuration."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import torch
from torch import Tensor


def rotate_polygons(polygons: Tensor, angles_rad: Tensor) -> Tensor:
    """Return ``polygons`` repeated and rotated by every supplied angle."""
    if polygons.ndim != 3 or polygons.shape[1:] != (4, 2):
        raise ValueError("polygons must have shape (N, 4, 2)")

    angles_rad = torch.as_tensor(
        angles_rad, dtype=polygons.dtype, device=polygons.device
    ).reshape(-1)
    cosine = torch.cos(angles_rad)
    sine = torch.sin(angles_rad)
    rotation_matrices = torch.stack(
        (cosine, -sine, sine, cosine), dim=1
    ).reshape(-1, 2, 2)

    # (P, N, 4, 2): one complete copy of the base panel for each angle.
    return torch.einsum("pij,nvj->pnvi", rotation_matrices, polygons).reshape(
        -1, 4, 2
    )


def generate_panel_detectors(
    *,
    detector_panel_inner_radius_mm: float,
    n_detector_panels: int,
    radial_layer_populations: Sequence[int],
    crystal_slot_tangential_mm: float,
    crystal_slot_radial_mm: float,
    crystal_tangential_mm: float,
    crystal_radial_mm: float,
) -> Tensor:
    """Generate the deterministic eight-layer SC-SPECT detector panels.

    A base panel is centered on the positive x-axis. Its x coordinate is
    radial and its y coordinate is tangential. Each radial layer contains a
    centered contiguous row of crystals, and the complete panel is repeated
    uniformly around the detector ring.

    ``detector_panel_inner_radius_mm`` locates the inner boundary of the first
    3.36-mm radial slot. Consequently, the first active crystal face is inset
    by half of the difference between slot depth and crystal depth.
    """
    if n_detector_panels <= 0:
        raise ValueError("n_detector_panels must be positive")
    if not radial_layer_populations or any(
        count <= 0 for count in radial_layer_populations
    ):
        raise ValueError("radial_layer_populations must contain positive counts")
    if crystal_tangential_mm > crystal_slot_tangential_mm:
        raise ValueError("crystal tangential size cannot exceed its slot pitch")
    if crystal_radial_mm > crystal_slot_radial_mm:
        raise ValueError("crystal radial size cannot exceed its slot pitch")

    crystals: list[Tensor] = []
    half_tangential = crystal_tangential_mm / 2.0
    half_radial = crystal_radial_mm / 2.0

    for layer_index, population in enumerate(radial_layer_populations):
        radial_center = detector_panel_inner_radius_mm + (
            layer_index + 0.5
        ) * crystal_slot_radial_mm
        tangential_centers = (
            torch.arange(population, dtype=torch.float32)
            - (population - 1) / 2.0
        ) * crystal_slot_tangential_mm

        layer = torch.empty((population, 4, 2), dtype=torch.float32)
        layer[:, 0, 0] = radial_center - half_radial
        layer[:, 0, 1] = tangential_centers - half_tangential
        layer[:, 1, 0] = radial_center + half_radial
        layer[:, 1, 1] = tangential_centers - half_tangential
        layer[:, 2, 0] = radial_center + half_radial
        layer[:, 2, 1] = tangential_centers + half_tangential
        layer[:, 3, 0] = radial_center - half_radial
        layer[:, 3, 1] = tangential_centers + half_tangential
        crystals.append(layer)

    base_panel = torch.cat(crystals, dim=0)
    panel_angles = torch.arange(n_detector_panels, dtype=torch.float32) * (
        2.0 * math.pi / n_detector_panels
    )
    return rotate_polygons(base_panel, panel_angles)


def generate_biconical_collimator(
    *,
    pinhole_diameter_mm: float,
    pinhole_opening_angle_deg: float,
    n_pinholes: int,
    collimator_ring_radius_mm: float,
    collimator_thickness_mm: float,
) -> Tensor:
    """Generate solid segments around focused biconical pinhole voids."""
    if n_pinholes <= 0:
        raise ValueError("n_pinholes must be positive")
    if pinhole_diameter_mm <= 0.0:
        raise ValueError("pinhole_diameter_mm must be positive")
    if not 0.0 < pinhole_opening_angle_deg < 180.0:
        raise ValueError("pinhole_opening_angle_deg must lie between 0 and 180")
    if collimator_thickness_mm <= 0.0:
        raise ValueError("collimator_thickness_mm must be positive")

    half_thickness = collimator_thickness_mm / 2.0
    junction_radius = collimator_ring_radius_mm
    radii = (
        junction_radius - half_thickness,
        junction_radius,
        junction_radius + half_thickness,
    )
    if radii[0] <= 0.0:
        raise ValueError("collimator thickness extends through the scanner origin")

    pinhole_pitch = 2.0 * math.pi / n_pinholes
    half_aperture = pinhole_diameter_mm / 2.0
    taper_offset = half_thickness * math.tan(
        math.radians(pinhole_opening_angle_deg) / 2.0
    )
    half_wide_opening = half_aperture + taper_offset

    inner_wide = math.asin(half_wide_opening / radii[0])
    junction_opening = math.asin(half_aperture / radii[1])
    outer_wide = math.asin(half_wide_opening / radii[2])
    if max(inner_wide, junction_opening, outer_wide) >= pinhole_pitch / 2.0:
        raise ValueError("adjacent pinhole openings overlap")

    inner_segments: list[list[list[float]]] = []
    outer_segments: list[list[list[float]]] = []
    for pinhole_index in range(n_pinholes):
        center = pinhole_index * pinhole_pitch
        cell_start = center - pinhole_pitch / 2.0
        cell_end = center + pinhole_pitch / 2.0

        inner_segments.append(
            _annular_segment_vertices(
                radii[0],
                radii[1],
                cell_start + inner_wide,
                cell_end - inner_wide,
                cell_start + junction_opening,
                cell_end - junction_opening,
            )
        )
        outer_segments.append(
            _annular_segment_vertices(
                radii[1],
                radii[2],
                cell_start + junction_opening,
                cell_end - junction_opening,
                cell_start + outer_wide,
                cell_end - outer_wide,
            )
        )

    return torch.tensor(inner_segments + outer_segments, dtype=torch.float32)


def _annular_segment_vertices(
    inner_radius: float,
    outer_radius: float,
    inner_start: float,
    inner_end: float,
    outer_start: float,
    outer_end: float,
) -> list[list[float]]:
    return [
        [inner_radius * math.cos(inner_start), inner_radius * math.sin(inner_start)],
        [outer_radius * math.cos(outer_start), outer_radius * math.sin(outer_start)],
        [outer_radius * math.cos(outer_end), outer_radius * math.sin(outer_end)],
        [inner_radius * math.cos(inner_end), inner_radius * math.sin(inner_end)],
    ]


def geometry_md5(*tensors: Tensor) -> str:
    """Return a stable identity hash for the generated geometry tensors."""
    digest = hashlib.md5()
    for value in tensors:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
