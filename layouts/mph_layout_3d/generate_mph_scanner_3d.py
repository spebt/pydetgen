# main_script_3d.py
import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from helper_3d import (
    generate_transaxial_spect_geometry_3d,
    generate_md5_from_tensors,
    plot_hexahedra_mpl3d,
    autoscale_3d_to_points,
    extrude_quads_to_hexahedra,
)

if __name__ == "__main__":
    # --- 1) Configuration (same physics, now with axial depths) ---
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
        "collimator_axial_mm": 20.0,   # axial depth of collimator ring (z extent)
        "detector_axial_mm": 6.0,      # axial depth of detector block (z extent)
        "z_center_mm": 0.0,
    }

    print("Generating 3D transaxial SPECT geometry (extruding 2D layout along z)...")
    geom3d = generate_transaxial_spect_geometry_3d(**cfg)

    det_hex = geom3d["detector_hexes"]
    in_hex  = geom3d["inner_coll_hexes"]
    out_hex = geom3d["outer_coll_hexes"]

    fov_size_xy_mm = 64.0
    fov_size_z_mm = 2.0
    h_xy = fov_size_xy_mm / 2.0
    h_z = fov_size_z_mm / 2.0

    # Define the 2D quad for the FOV
    fov_quad_2d = torch.tensor([[
        [-h_xy, -h_xy], [h_xy, -h_xy], [h_xy, h_xy], [-h_xy, h_xy]
    ]], dtype=torch.float32)

    # Extrude it to 3D
    fov_hex = extrude_quads_to_hexahedra(fov_quad_2d, z_min=-h_z, z_max=h_z)

    # --- 2) (Optional) Save a single 3D layout file like your 2D tensor ---
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(_script_dir, "..", "..", "..", "data", "scanner_layouts")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "mph_hourglass_single_position_base_3d_v2.tensor")
    all_plate_hex = torch.cat([in_hex, out_hex], dim=0)
    # all_plate_hex = torch.zeros((0, 8, 3), dtype=det_hex.dtype)#torch.cat([in_hex, out_hex], dim=0) <-- changed this to remove plate
    scanner_md5 = generate_md5_from_tensors(det_hex, all_plate_hex)
    torch.save({
        "scanner MD5": scanner_md5,
        "applied_config": cfg,
        "layouts": {
            "position 000": {
                "position": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                "detector units 3d": det_hex,                   # (Nd,8,3)
                "plate segments 3d": all_plate_hex,             # (Ni+No, 8, 3)
            }
        }
    }, out_file)
    print(f"Saved 3D layout to:\n  {out_file}")

    # --- 3) Plot in 3D ---
    print("Plotting 3D scanner layout...")
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Collimator (inner + outer)
    if in_hex.numel() > 0:
        plot_hexahedra_mpl3d(in_hex, ax, facecolor="gray", edgecolor="black", alpha=0.35, wireframe=False)
    if out_hex.numel() > 0:
        plot_hexahedra_mpl3d(out_hex, ax, facecolor="lightgray", edgecolor="black", alpha=0.35, wireframe=False)

    # Detectors
    if det_hex.numel() > 0:
        plot_hexahedra_mpl3d(det_hex, ax, facecolor="skyblue", edgecolor="navy", alpha=0.5, wireframe=False)

    # Plot FOV
    if fov_hex.numel() > 0:
        print("Plotting FOV box...")
        plot_hexahedra_mpl3d(
            fov_hex, 
            ax, 
            facecolor="magenta", 
            edgecolor="magenta", 
            alpha=0.1, 
            wireframe=False
        )
    # Camera/axes framing
    all_points_for_scaling = [det_hex, in_hex, out_hex, fov_hex]
    valid_points_for_scaling = [p for p in all_points_for_scaling if p.numel() > 0]
    autoscale_3d_to_points(ax, *valid_points_for_scaling, margin=0.08)
    ax.set_title("SPECT Scanner Base Layout (3D)")

    plot_filename = "spect_base_layout_3d.png"
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    print(f"3D plot saved to {plot_filename}")
    plt.show()
