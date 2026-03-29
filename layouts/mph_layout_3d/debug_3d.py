# %% [markdown]
# # SPECT 2D/3D Geometry Debug Notebook
#
# This notebook is meant for **visual + numeric debugging** of:
# - 2D transaxial geometry (detectors + inner/outer collimator segments)
# - 3D extrusion into hexahedra
# - FOV placement
# - Basic coverage diagnostics (crystal packing, radii, etc.)
#
# It assumes:
# - helper.py
# - helper_3d.py
# live in the same directory.

# %%
import os
import math
from typing import Tuple

import torch
from torch import Tensor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from helper import (
    generate_transaxial_spect_geometry,
    translate_vertices_2d,
    make_transaxial_positions,
)
from helper_3d import (
    generate_transaxial_spect_geometry_3d,
    generate_md5_from_tensors,
    extrude_quads_to_hexahedra,
    plot_hexahedra_mpl3d,
    autoscale_3d_to_points,
)



plt.rcParams["figure.figsize"] = (7, 7)
plt.rcParams["axes.grid"] = True

device = torch.device("cpu")
dtype = torch.float32

# %% [markdown]
# ## 1. Global Configuration
#
# Adjust these values to match `generate_mph_scanner_3d.py`.

# %%
cfg = {
    "pinhole_diameter_mm": 2.0,
    "pinhole_opening_angle_deg": 27.0,
    "n_pinholes": 18,
    "collimator_ring_radius_mm": 215.0,
    "collimator_thickness_mm": 20.0,
    "detector_ring_radius_mm": 215.0 + 542.0,
    "detector_thickness_mm": 6.0,
    "detector_crystal_width_mm": 3.5,
    # 3D extras:
    "collimator_axial_mm": 20.0,
    "detector_axial_mm": 6.0,
    "z_center_mm": 0.0,
}

fov_size_xy_mm = 64.0
fov_size_z_mm = 2.0

# %% [markdown]
# ## 2. Utility Functions for Diagnostics

# %%
def polygon_centroids_2d(polys: Tensor) -> Tensor:
    """
    polys: (N,4,2)
    returns centroids: (N,2)
    """
    return polys.mean(dim=1)


def polygon_angles_2d(polys: Tensor) -> Tensor:
    """
    Approximate angular position of each polygon as the angle
    of its centroid relative to origin.
    polys: (N,4,2)
    returns: (N,) angles in radians in [-pi, pi]
    """
    c = polygon_centroids_2d(polys)
    return torch.atan2(c[:, 1], c[:, 0])


def print_basic_2d_stats(det: Tensor, inner: Tensor, outer: Tensor):
    print("=== Basic 2D Geometry Stats ===")
    print(f"Detectors:               {det.shape[0]} quads")
    print(f"Inner collimator segs:   {inner.shape[0]} quads")
    print(f"Outer collimator segs:   {outer.shape[0]} quads")

    # Radii ranges
    def radii_range(polys: Tensor, name: str):
        r = torch.linalg.norm(polys.view(-1, 2), dim=1)
        print(f"{name:25s} radius min/max: {r.min():.2f} mm / {r.max():.2f} mm")

    radii_range(det, "Detectors")
    radii_range(inner, "Inner collimator")
    radii_range(outer, "Outer collimator")

    print()


def plot_2d_ring_layout(
    det: Tensor,
    inner: Tensor,
    outer: Tensor,
    fov_size_xy: float = 64.0,
    title: str = "2D Ring Layout (Top View)",
):
    """
    Simple 2D top view: detectors + inner + outer collimator + FOV box.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    def draw_quads(polys: Tensor, color: str, alpha: float, label: str):
        for i in range(polys.shape[0]):
            xy = polys[i].detach().cpu()
            xs = xy[:, 0].tolist() + [xy[0, 0].item()]
            ys = xy[:, 1].tolist() + [xy[0, 1].item()]
            ax.plot(xs, ys, color=color, alpha=alpha)
        # add one dummy point for legend
        ax.plot([], [], color=color, alpha=alpha, label=label)

    if inner.numel() > 0:
        draw_quads(inner, "tab:orange", 0.7, "Inner collimator segments")
    if outer.numel() > 0:
        draw_quads(outer, "tab:red", 0.7, "Outer collimator segments")
    if det.numel() > 0:
        draw_quads(det, "tab:blue", 0.6, "Detector crystals")

    # FOV box for visual sanity
    h = fov_size_xy / 2.0
    fov_box = torch.tensor(
        [[[-h, -h], [h, -h], [h, h], [-h, h]]], dtype=torch.float32
    )

    fb = fov_box[0]
    xs = fb[:, 0].tolist() + [fb[0, 0].item()]
    ys = fb[:, 1].tolist() + [fb[0, 1].item()]
    ax.plot(xs, ys, "m--", linewidth=1.5, label=f"FOV {fov_size_xy} mm")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True)
    # plt.show()
    plt.savefig("debug/2d_ring_layout.png")


def plot_segment_angles(inner: Tensor, outer: Tensor, n_pinholes: int):
    """
    Visual diagnostics: show angular distribution of inner/outer collimator segments.
    """
    inner_ang = polygon_angles_2d(inner)
    outer_ang = polygon_angles_2d(outer)

    inner_deg = inner_ang * 180.0 / math.pi
    outer_deg = outer_ang * 180.0 / math.pi

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(inner_deg, torch.zeros_like(inner_deg), label="Inner segments", marker="o")
    ax.scatter(outer_deg, torch.ones_like(outer_deg), label="Outer segments", marker="x")

    # Also draw the pinhole centers at ideal angular spacing
    pinhole_deg = torch.linspace(0, 360, steps=n_pinholes + 1)[:-1]
    ax.vlines(pinhole_deg, ymin=-0.5, ymax=1.5, colors="gray", linestyles="dotted", alpha=0.5, label="Pinhole centers")

    ax.set_ylim(-1.0, 2.0)
    ax.set_yticks([])
    ax.set_xlabel("Angle (deg)")
    ax.set_title("Angular positions of collimator segments")
    ax.legend(loc="upper right")
    ax.grid(True)
    # plt.show()
    plt.savefig("debug/segment_angles_plot.png")


def compute_crystal_spacing_stats(
    detector_ring_radius_mm: float,
    detector_thickness_mm: float,
    detector_crystal_width_mm: float,
):
    """
    Recompute expected crystal count and angular spacing,
    matching the logic in helper.generate_transaxial_spect_geometry (for sanity).
    """
    R_DET_INNER = detector_ring_radius_mm
    mid_radius_det = R_DET_INNER + detector_thickness_mm / 2.0
    circumference_det = 2 * math.pi * mid_radius_det
    n_total_crystals = int(circumference_det / detector_crystal_width_mm)
    angular_width_crystal = detector_crystal_width_mm / mid_radius_det
    total_crystal_angular_span = n_total_crystals * angular_width_crystal
    total_angular_gap = 2 * math.pi - total_crystal_angular_span
    angular_gap_between_crystals = total_angular_gap / n_total_crystals
    angle_step_per_crystal = angular_width_crystal + angular_gap_between_crystals

    print("=== Crystal Packing (Expected) ===")
    print(f"Mid radius:                     {mid_radius_det:.2f} mm")
    print(f"Detector circumference:         {circumference_det:.2f} mm")
    print(f"Crystal width:                  {detector_crystal_width_mm:.2f} mm")
    print(f"Estimated number of crystals:   {n_total_crystals}")
    print(
        "Angular width per crystal:      "
        f"{math.degrees(angular_width_crystal):.4f} deg"
    )
    print(
        "Angular gap between crystals:   "
        f"{math.degrees(angular_gap_between_crystals):.4f} deg"
    )
    print(
        "Angular step (width+gap):       "
        f"{math.degrees(angle_step_per_crystal):.4f} deg"
    )
    print()


def build_fov_hex(fov_xy_mm: float, fov_z_mm: float) -> Tensor:
    """
    Build 3D FOV hexahedron using same convention as in generate_mph_scanner_3d.py
    """
    h_xy = fov_xy_mm / 2.0
    h_z = fov_z_mm / 2.0

    # 2D quad (1,4,2)
    fov_quad_2d = torch.tensor(
        [[[-h_xy, -h_xy], [h_xy, -h_xy], [h_xy, h_xy], [-h_xy, h_xy]]],
        dtype=torch.float32,
    )
    fov_hex = extrude_quads_to_hexahedra(fov_quad_2d, z_min=-h_z, z_max=h_z)
    return fov_hex


# %% [markdown]
# ## 3. Generate 2D Geometry and Run Basic Checks

# %%
with torch.no_grad():
    det_2d, inner_2d, outer_2d = generate_transaxial_spect_geometry(
        pinhole_diameter_mm=cfg["pinhole_diameter_mm"],
        pinhole_opening_angle_deg=cfg["pinhole_opening_angle_deg"],
        n_pinholes=cfg["n_pinholes"],
        collimator_ring_radius_mm=cfg["collimator_ring_radius_mm"],
        collimator_thickness_mm=cfg["collimator_thickness_mm"],
        detector_ring_radius_mm=cfg["detector_ring_radius_mm"],
        detector_thickness_mm=cfg["detector_thickness_mm"],
        detector_crystal_width_mm=cfg["detector_crystal_width_mm"],
    )

print_basic_2d_stats(det_2d, inner_2d, outer_2d)
compute_crystal_spacing_stats(
    detector_ring_radius_mm=cfg["detector_ring_radius_mm"],
    detector_thickness_mm=cfg["detector_thickness_mm"],
    detector_crystal_width_mm=cfg["detector_crystal_width_mm"],
)

plot_2d_ring_layout(
    det_2d, inner_2d, outer_2d,
    fov_size_xy=fov_size_xy_mm,
    title="2D Layout: Detectors + Double-Taper Collimator + FOV",
)

plot_segment_angles(inner_2d, outer_2d, n_pinholes=cfg["n_pinholes"])

# %% [markdown]
# ### 3.1 Optional: Focus on a Single Pinhole Region
#
# This cell zooms in on a small angular window to visually inspect segment alignment.

# %%
def plot_zoomed_region(
    det: Tensor,
    inner: Tensor,
    outer: Tensor,
    center_angle_deg: float,
    window_deg: float = 30.0,
    title: str = "Zoomed Region Around a Pinhole",
):
    center_rad = math.radians(center_angle_deg)
    window_rad = math.radians(window_deg) / 2.0

    def keep(polys: Tensor) -> Tensor:
        ang = polygon_angles_2d(polys)
        # bring into principal value around center
        diff = (ang - center_rad + math.pi) % (2 * math.pi) - math.pi
        mask = (diff >= -window_rad) & (diff <= window_rad)
        return polys[mask]

    det_z = keep(det)
    inner_z = keep(inner)
    outer_z = keep(outer)

    title2 = f"{title}\ncenter={center_angle_deg:.1f} deg, window={window_deg:.1f} deg"
    plot_2d_ring_layout(det_z, inner_z, outer_z, fov_size_xy=64.0, title=title2)


# Zoom around angle = 0 deg (x-axis)
plot_zoomed_region(det_2d, inner_2d, outer_2d, center_angle_deg=0.0, window_deg=40.0)

# %% [markdown]
# ## 4. Generate 3D Geometry (Hexahedra) and Visualize
#
# We reuse the existing 3D generator from `helper_3d.py` and add:
# - FOV box in 3D
# - Autoscaled camera framing
# - Simple z-extent checks

# %%
with torch.no_grad():
    geom3d = generate_transaxial_spect_geometry_3d(**cfg)

det_hex = geom3d["detector_hexes"]
in_hex = geom3d["inner_coll_hexes"]
out_hex = geom3d["outer_coll_hexes"]

print("=== 3D Hex Counts ===")
print(f"Detector hexes:         {det_hex.shape[0]}")
print(f"Inner collimator hexes: {in_hex.shape[0]}")
print(f"Outer collimator hexes: {out_hex.shape[0]}")
print()

# Build FOV hex
fov_hex = build_fov_hex(fov_size_xy_mm, fov_size_z_mm)

# Z-range diagnostics
def print_z_range(name: str, hexes: Tensor):
    if hexes.numel() == 0:
        print(f"{name:25s} z-range: (empty)")
        return
    z = hexes[..., 2]
    print(f"{name:25s} z-range: {z.min():.2f} mm .. {z.max():.2f} mm")

print_z_range("Detector hexes", det_hex)
print_z_range("Inner collimator hexes", in_hex)
print_z_range("Outer collimator hexes", out_hex)
print_z_range("FOV hex", fov_hex)

# %%
# 3D visualization of scanner + FOV

print("Plotting 3D scanner layout...")

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection="3d")

if in_hex.numel() > 0:
    plot_hexahedra_mpl3d(in_hex, ax, facecolor="gray", edgecolor="black", alpha=0.35, wireframe=False)
if out_hex.numel() > 0:
    plot_hexahedra_mpl3d(out_hex, ax, facecolor="lightgray", edgecolor="black", alpha=0.35, wireframe=False)
if det_hex.numel() > 0:
    plot_hexahedra_mpl3d(det_hex, ax, facecolor="skyblue", edgecolor="navy", alpha=0.5, wireframe=False)

# FOV in magenta
if fov_hex.numel() > 0:
    plot_hexahedra_mpl3d(
        fov_hex,
        ax,
        facecolor="magenta",
        edgecolor="magenta",
        alpha=0.1,
        wireframe=False,
    )

autoscale_3d_to_points(ax, det_hex, in_hex, out_hex, fov_hex, margin=0.08)
ax.set_title("SPECT Scanner Base Layout (3D)")

plt.tight_layout()
# plt.show()
plt.savefig("debug/layout.png")

# %% [markdown]
# ## 5. Slice Views in 3D (Optional)
#
# This section gives you a simple way to inspect cross-sections
# by thresholding on |z|.

# %%
def extract_slice(hexes: Tensor, z_min: float, z_max: float) -> Tensor:
    """
    Keep only hexes whose centroid z is in [z_min, z_max].
    """
    if hexes.numel() == 0:
        return hexes
    z_centroid = hexes[..., 2].mean(dim=1)
    mask = (z_centroid >= z_min) & (z_centroid <= z_max)
    return hexes[mask]


def plot_slice(det_hex: Tensor, in_hex: Tensor, out_hex: Tensor, z_min: float, z_max: float):
    det_s = extract_slice(det_hex, z_min, z_max)
    in_s = extract_slice(in_hex, z_min, z_max)
    out_s = extract_slice(out_hex, z_min, z_max)

    print(f"Slice z in [{z_min:.2f}, {z_max:.2f}] mm")
    print(f" Detectors in slice:   {det_s.shape[0]}")
    print(f" Inner collimator:     {in_s.shape[0]}")
    print(f" Outer collimator:     {out_s.shape[0]}")

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    if in_s.numel() > 0:
        plot_hexahedra_mpl3d(in_s, ax, facecolor="gray", edgecolor="black", alpha=0.5, wireframe=False)
    if out_s.numel() > 0:
        plot_hexahedra_mpl3d(out_s, ax, facecolor="lightgray", edgecolor="black", alpha=0.5, wireframe=False)
    if det_s.numel() > 0:
        plot_hexahedra_mpl3d(det_s, ax, facecolor="skyblue", edgecolor="navy", alpha=0.8, wireframe=False)

    autoscale_3d_to_points(ax, det_s, in_s, out_s, margin=0.05)
    ax.set_title(f"Slice view: z in [{z_min:.1f}, {z_max:.1f}] mm")
    plt.tight_layout()
    # plt.show()
    plt.savefig("debug/plot_slice.png")


# Example: near isocenter slice
plot_slice(det_hex, in_hex, out_hex, z_min=-2.0, z_max=2.0)

# %% [markdown]
# ## 6. Save Layout Tensor (Compatibility with main_script_3d)
#
# This mimics `generate_mph_scanner_3d.py` file output so you can check
# that what you debug here matches the data you feed into the simulator.

# %%
_script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(_script_dir, "..", "..", "..", "data", "scanner_layouts")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "dbg_mph_hourglass_single_position_base_3d_v2.tensor")

all_plate_hex = torch.cat([in_hex, out_hex], dim=0)
scanner_md5 = generate_md5_from_tensors(det_hex, all_plate_hex)

torch.save(
    {
        "scanner MD5": scanner_md5,
        "applied_config": cfg,
        "layouts": {
            "position 000": {
                "position": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                "detector units 3d": det_hex,
                "plate segments 3d": all_plate_hex,
            }
        },
    },
    out_file,
)
print(f"Saved 3D layout tensor to:\n  {out_file}")

# %% [markdown]
# ## 7. Quick Numeric Sanity Checks
#
# These give compact numbers you can track as you change geometry.

# %%
def quick_numeric_checks(det: Tensor, inner: Tensor, outer: Tensor):
    """
    Some extra scalar checks: angular coverage, radial gaps, etc.
    """
    print("=== Quick Numeric Checks (2D) ===")
    # Angular coverage of detectors (based on centroids)
    det_ang = polygon_angles_2d(det)
    det_ang_sorted, _ = torch.sort(det_ang)
    # spacing between consecutive centroids (wrap around)
    diffs = torch.diff(det_ang_sorted)
    wrap = (det_ang_sorted[0] + 2 * math.pi) - det_ang_sorted[-1]
    diffs = torch.cat([diffs, wrap.unsqueeze(0)])
    print(
        "Detector centroid angular spacing: "
        f"min={math.degrees(diffs.min()):.4f} deg, "
        f"max={math.degrees(diffs.max()):.4f} deg, "
        f"mean={math.degrees(diffs.mean()):.4f} deg"
    )

    # Collimator segment angular densities
    for name, seg in [("Inner coll", inner), ("Outer coll", outer)]:
        if seg.numel() == 0:
            continue
        ang = polygon_angles_2d(seg)
        ang_sorted, _ = torch.sort(ang)
        diffs = torch.diff(ang_sorted)
        wrap = (ang_sorted[0] + 2 * math.pi) - ang_sorted[-1]
        diffs = torch.cat([diffs, wrap.unsqueeze(0)])
        print(
            f"{name:10s} spacing: "
            f"min={math.degrees(diffs.min()):.4f} deg, "
            f"max={math.degrees(diffs.max()):.4f} deg, "
            f"mean={math.degrees(diffs.mean()):.4f} deg"
        )

    print()

quick_numeric_checks(det_2d, inner_2d, outer_2d)
