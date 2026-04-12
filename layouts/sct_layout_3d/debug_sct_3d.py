# debug_sct_3d.py
#
# Interactive debug / validation script for the SCT flat-panel geometry.
# Run as a plain Python script.  Saves plots to debug/ and a debug .tensor
# file to data/scanner_layouts/ so you can diff it against the main output.
#
# Checks performed:
#   1. Crystal counts per layer and bounding-box sanity
#   2. Y-range of each detector layer (no overlaps)
#   3. Hole placement: count, spacing, edge clearance
#   4. 3D visualisation (top-down XZ view + full 3D view)
#   5. Layer-by-layer XZ footprints
#   6. Saves debug .tensor for comparison

import os
import math
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from helper_sct_3d import (
    generate_sct_detector_hexes,
    generate_sct_collimator_hex,
    generate_sct_collimator_holes,
    generate_md5_from_tensors,
    plot_hexahedra_mpl3d,
    autoscale_3d_to_points,
    box_to_hex,
)

os.makedirs("debug", exist_ok=True)

plt.rcParams["figure.figsize"] = (8, 7)
plt.rcParams["axes.grid"] = True

# -----------------------------------------------------------------------
# Configuration — keep in sync with generate_sct_scanner_3d.py
# -----------------------------------------------------------------------
cfg = {
    "fov_to_collimator_front_mm": 67.0,
    "collimator_width_x_mm":     140.0,
    "collimator_thickness_y_mm":   6.0,
    "collimator_height_z_mm":    140.0,
    "hole_radius_mm":              0.8,
    "num_holes":                  1218,
    "hole_seed":                    42,
    "detector_gap_from_collimator_mm": 10.0,
    "detector_layer_gap_mm": 2.0,
    "detector_layers": [
        {"nx": 32, "nz": 16, "crystal_x_mm": 3.0, "crystal_y_mm": 6.0, "crystal_z_mm": 3.0, "pitch_x_mm": 4.0, "pitch_z_mm": 8.0},
        {"nx": 32, "nz": 16, "crystal_x_mm": 3.0, "crystal_y_mm": 6.0, "crystal_z_mm": 3.0, "pitch_x_mm": 4.0, "pitch_z_mm": 8.0},
        {"nx": 32, "nz": 16, "crystal_x_mm": 3.0, "crystal_y_mm": 6.0, "crystal_z_mm": 3.0, "pitch_x_mm": 4.0, "pitch_z_mm": 8.0},
        {"nx": 64, "nz": 64, "crystal_x_mm": 2.0, "crystal_y_mm": 6.0, "crystal_z_mm": 2.0, "pitch_x_mm": 2.0, "pitch_z_mm": 2.0},
    ],
}

collimator_y_center = cfg["fov_to_collimator_front_mm"] + cfg["collimator_thickness_y_mm"] / 2.0
collimator_back_y   = cfg["fov_to_collimator_front_mm"] + cfg["collimator_thickness_y_mm"]
cfg["collimator_y_center_mm"] = collimator_y_center
cfg["detector_y_start_mm"]    = collimator_back_y + cfg["detector_gap_from_collimator_mm"]

# -----------------------------------------------------------------------
# Generate geometry
# -----------------------------------------------------------------------
det_hex   = generate_sct_detector_hexes(cfg)
plate_hex = generate_sct_collimator_hex(cfg)
holes     = generate_sct_collimator_holes(cfg, seed=cfg["hole_seed"])

fov_half = 64.0
fov_hex  = box_to_hex(0.0, 0.0, 0.0, 2 * fov_half, 2 * fov_half, 2 * fov_half).unsqueeze(0)

# -----------------------------------------------------------------------
# 1. Crystal count and Y-range checks
# -----------------------------------------------------------------------
print("=== Detector Crystal Counts ===")
expected_total = sum(l["nx"] * l["nz"] for l in cfg["detector_layers"])
print(f"Expected total crystals : {expected_total}")
print(f"Generated hexes         : {det_hex.shape[0]}")
assert det_hex.shape[0] == expected_total, "Crystal count mismatch!"

print("\n=== Detector Layer Y-Ranges (world coordinates) ===")
offset = 0
current_y = cfg["detector_y_start_mm"]
gap_y = cfg["detector_layer_gap_mm"]
for li, layer in enumerate(cfg["detector_layers"]):
    n = layer["nx"] * layer["nz"]
    dy = layer["crystal_y_mm"]
    layer_hexes = det_hex[offset : offset + n]
    y_vals = layer_hexes[..., 1]  # all Y coordinates for this layer
    y_front = y_vals.min().item()
    y_back  = y_vals.max().item()
    expected_front = current_y
    expected_back  = current_y + dy
    label = "Mosaic" if li < 3 else "HR    "
    print(
        f"  Layer {li} ({label}, {layer['nx']}x{layer['nz']}): "
        f"Y [{y_front:.2f}, {y_back:.2f}] mm  "
        f"(expected [{expected_front:.2f}, {expected_back:.2f}])"
    )
    offset += n
    current_y += dy + gap_y

print(f"\nCollimator front face Y : {cfg['fov_to_collimator_front_mm']:.2f} mm")
print(f"Collimator centre Y     : {collimator_y_center:.2f} mm")
print(f"Collimator back face Y  : {collimator_back_y:.2f} mm")
print(f"Detector stack start Y  : {cfg['detector_y_start_mm']:.2f} mm")
print(f"Detector stack end   Y  : {det_hex[..., 1].max().item():.2f} mm")

# -----------------------------------------------------------------------
# 2. Hole placement checks
# -----------------------------------------------------------------------
print("\n=== Collimator Hole Checks ===")
print(f"Holes generated : {holes.shape[0]} / {cfg['num_holes']} requested")

hole_x = holes[:, 0]
hole_z = holes[:, 3]
r = cfg["hole_radius_mm"]
half_x = cfg["collimator_width_x_mm"]  / 2.0
half_z = cfg["collimator_height_z_mm"] / 2.0

edge_x = (hole_x.abs() + r).max().item()
edge_z = (hole_z.abs() + r).max().item()
print(f"Max |x| + r     : {edge_x:.3f} mm  (limit {half_x:.1f} mm)")
print(f"Max |z| + r     : {edge_z:.3f} mm  (limit {half_z:.1f} mm)")
assert edge_x <= half_x + 1e-4, "Hole out of X bounds!"
assert edge_z <= half_z + 1e-4, "Hole out of Z bounds!"

# Minimum centre-to-centre distance (sample check on first 200 holes)
n_check = min(200, holes.shape[0])
hx_s = hole_x[:n_check].unsqueeze(1)  # (n_check, 1)
hz_s = hole_z[:n_check].unsqueeze(1)
dist2 = (hx_s - hx_s.T) ** 2 + (hz_s - hz_s.T) ** 2
dist2.fill_diagonal_(float("inf"))
min_dist = dist2.min().sqrt().item()
print(f"Min hole-to-hole distance (first {n_check}): {min_dist:.3f} mm  (min allowed {2*r:.2f} mm)")

open_area_approx = holes.shape[0] * math.pi * r**2
plate_area = cfg["collimator_width_x_mm"] * cfg["collimator_height_z_mm"]
print(f"Approx open area fraction: {open_area_approx / plate_area * 100:.1f}%")

# -----------------------------------------------------------------------
# 3. XZ hole map (top-down view of collimator face)
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
for i in range(holes.shape[0]):
    circle = plt.Circle((holes[i, 0].item(), holes[i, 3].item()), r, color="white", linewidth=0.3, edgecolor="gray")
    ax.add_patch(circle)
ax.set_xlim(-half_x, half_x)
ax.set_ylim(-half_z, half_z)
ax.set_facecolor("dimgray")
ax.set_aspect("equal")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Z (mm)")
ax.set_title(f"Collimator Hole Map (XZ, top-down)\n{holes.shape[0]} holes, r={r} mm")
plt.tight_layout()
plt.savefig("debug/sct_collimator_holes_xz.png", dpi=150)
print("\nSaved: debug/sct_collimator_holes_xz.png")

# -----------------------------------------------------------------------
# 4. Layer footprints (XZ view of each detector layer)
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
offset = 0
for li, layer in enumerate(cfg["detector_layers"]):
    n = layer["nx"] * layer["nz"]
    layer_hexes = det_hex[offset : offset + n]  # (n, 8, 3)
    cx = layer_hexes[:, :4, 0].mean(dim=1).numpy()  # crystal centres in X
    cz = layer_hexes[:, :4, 2].mean(dim=1).numpy()  # crystal centres in Z
    axes[li].scatter(cx, cz, s=1.5, alpha=0.6)
    label = "Mosaic" if li < 3 else "HR"
    axes[li].set_title(f"Layer {li} ({label})\n{layer['nx']}×{layer['nz']} crystals")
    axes[li].set_xlabel("X (mm)")
    axes[li].set_ylabel("Z (mm)")
    axes[li].set_aspect("equal")
    offset += n
plt.suptitle("Detector Crystal Footprints by Layer (XZ plane)")
plt.tight_layout()
plt.savefig("debug/sct_detector_layers_xz.png", dpi=150)
print("Saved: debug/sct_detector_layers_xz.png")

# -----------------------------------------------------------------------
# 5. Full 3D visualisation
# -----------------------------------------------------------------------
print("\nPlotting 3D scanner layout ...")
fig3 = plt.figure(figsize=(10, 9))
ax3 = fig3.add_subplot(111, projection="3d")

plot_hexahedra_mpl3d(plate_hex, ax3, facecolor="dimgray", edgecolor="black", alpha=0.35)

layer_colours = ["steelblue", "dodgerblue", "deepskyblue", "gold"]
offset = 0
for li, layer in enumerate(cfg["detector_layers"]):
    n = layer["nx"] * layer["nz"]
    plot_hexahedra_mpl3d(
        det_hex[offset : offset + n],
        ax3,
        facecolor=layer_colours[li],
        edgecolor="navy",
        alpha=0.5,
    )
    offset += n

plot_hexahedra_mpl3d(fov_hex, ax3, facecolor="magenta", edgecolor="magenta", alpha=0.07)

autoscale_3d_to_points(ax3, det_hex, plate_hex, margin=0.06)
ax3.set_title("SCT Scanner Debug Layout (3D)")
plt.tight_layout()
plt.savefig("debug/sct_layout_3d.png", dpi=150)
print("Saved: debug/sct_layout_3d.png")

# -----------------------------------------------------------------------
# 6. Save debug .tensor (same structure as generate_sct_scanner_3d.py)
# -----------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(_script_dir, "..", "..", "..", "data", "scanner_layouts")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "dbg_sct_single_position_base_3d.tensor")

scanner_md5 = generate_md5_from_tensors(det_hex, plate_hex)
torch.save(
    {
        "scanner MD5": scanner_md5,
        "applied_config": cfg,
        "collimator holes": holes,
        "layouts": {
            "position 000": {
                "position": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                "detector units 3d": det_hex,
                "plate segments 3d": plate_hex,
            }
        },
    },
    out_file,
)
print(f"\nSaved debug .tensor to:\n  {out_file}")
print(f"  scanner MD5 : {scanner_md5}")
print("\nDone.")
