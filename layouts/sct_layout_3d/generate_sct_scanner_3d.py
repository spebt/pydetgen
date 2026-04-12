# generate_sct_scanner_3d.py
#
# Entry point: generates the SCT flat-panel SPECT scanner geometry as a 3D
# hexahedral .tensor file compatible with the pymatcal pipeline.
#
# Geometry:
#   - Collimator: single flat plate (140x6x140 mm) with 1218 random cylindrical
#     holes (r=0.8 mm).  The plate is stored as 1 hexahedron in 'plate segments 3d';
#     hole positions are stored separately under 'collimator holes'.
#   - Detector: 4-layer flat-panel array.
#       Layers 0-2 (Mosaic):  32×16 crystals, 3×6×3 mm, pitch 4×8 mm
#       Layer  3   (HR):      64×64 crystals, 2×6×2 mm, pitch 2×2 mm
#       Total: 5632 crystals
#
# Coordinate system (FOV centre at origin):
#   X: transverse (left-right)
#   Y: depth (increasing away from FOV → collimator → detector)
#   Z: axial   (up-down)
#
# Output: data/scanner_layouts/sct_single_position_base_3d.tensor

import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from helper_sct_3d import (
    generate_sct_detector_hexes,
    generate_sct_collimator_hex,
    generate_sct_collimator_holes,
    generate_sct_hole_prisms,
    generate_md5_from_tensors,
    plot_hexahedra_mpl3d,
    autoscale_3d_to_points,
    box_to_hex,
)

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # 1) Configuration
    #
    #  All values are component sizes or air gaps — no world-coordinate math
    #  needed here.  Y increases away from the patient (toward the detector).
    #  FOV sizes (x/y/z) are for VISUALISATION only — changing them does NOT
    #  move any hardware.  Hardware positions are fixed by the gap parameters
    #  below, all measured from the FOV centre (world origin, y = 0).
    #
    #  Physical chain along Y (all gaps from FOV centre):
    #
    #    [FOV centre / world origin  y = 0]
    #        + fov_to_collimator_gap_mm       → collimator entrance face
    #        + collimator_thickness_y_mm      → collimator exit face
    #        + collimator_to_detector_gap_mm  → detector layer 0 front face
    #        + crystal_y_mm + detector_layer_gap_mm  → layer 1 front face …
    # -----------------------------------------------------------------------
    cfg = {
        # --- Imaging volume (FOV) — VISUALISATION only, does not affect hardware positions ---
        "fov_size_x_mm":  32.0,   # transverse width  (X)
        "fov_size_y_mm":  32.0,   # depth             (Y)
        "fov_size_z_mm":  32.0,   # axial height      (Z)

        # --- Hardware positions (measured from FOV centre = world origin) ---
        "fov_to_collimator_gap_mm":  100.0,  # FOV centre → collimator entrance face

        # --- Collimator ---
        "collimator_width_x_mm":     140.0,   # plate extent in X
        "collimator_thickness_y_mm":   6.0,   # plate depth (Y)
        "collimator_height_z_mm":    140.0,   # plate extent in Z
        "hole_radius_mm":              0.8,   # cylindrical hole radius
        "num_holes":                  1218,   # number of holes (≈12.5% open area)
        "hole_seed":                    42,   # RNG seed for reproducible placement

        # --- Material attenuation coefficients (mm^-1 at 140 keV) ---
        # Stored here so pymatcal can read them from applied_config instead of hardcoding.
        "mu_plate_mm_inv":    4.0,   # tungsten collimator (Param_Collimator.py)
        "mu_detector_mm_inv": 0.35,  # detector crystal    (Param_Detector.py)

        # --- Air gap: collimator exit face → detector layer 0 front face ---
        "collimator_to_detector_gap_mm": 10.0,

        # --- Detector stack ---
        # Gap between the exit face of one layer and the entrance face of the next.
        "detector_layer_gap_mm": 2.0,
        # Layer definitions: nx (columns in X), nz (rows in Z),
        # crystal dimensions (mm), pitch (centre-to-centre, mm).
        "detector_layers": [
            # --- Mosaic layers 0, 1, 2 ---
            {
                "nx": 32, "nz": 16,
                "crystal_x_mm": 3.0, "crystal_y_mm": 6.0, "crystal_z_mm": 3.0,
                "pitch_x_mm": 4.0,   "pitch_z_mm": 8.0,
            },
            {
                "nx": 32, "nz": 16,
                "crystal_x_mm": 3.0, "crystal_y_mm": 6.0, "crystal_z_mm": 3.0,
                "pitch_x_mm": 4.0,   "pitch_z_mm": 8.0,
            },
            {
                "nx": 32, "nz": 16,
                "crystal_x_mm": 3.0, "crystal_y_mm": 6.0, "crystal_z_mm": 3.0,
                "pitch_x_mm": 4.0,   "pitch_z_mm": 8.0,
            },
            # --- High-resolution layer 3 ---
            {
                "nx": 64, "nz": 64,
                "crystal_x_mm": 2.0, "crystal_y_mm": 6.0, "crystal_z_mm": 2.0,
                "pitch_x_mm": 2.0,   "pitch_z_mm": 2.0,
            },
        ],
    }

    # -----------------------------------------------------------------------
    # 2) Derive world-space Y positions from the physical chain above.
    #    Edit cfg values above, not these derived variables.
    # -----------------------------------------------------------------------
    collimator_front_y = cfg["fov_to_collimator_gap_mm"]
    collimator_back_y  = collimator_front_y + cfg["collimator_thickness_y_mm"]
    detector_y_start   = collimator_back_y  + cfg["collimator_to_detector_gap_mm"]

    # Inject derived values so helpers have a single flat cfg to read from
    cfg["collimator_y_center_mm"] = (collimator_front_y + collimator_back_y) / 2.0
    cfg["detector_y_start_mm"]    = detector_y_start

    # -----------------------------------------------------------------------
    # 3) Generate geometry
    # -----------------------------------------------------------------------
    print("Generating SCT flat-panel detector hexahedra ...")
    det_hex = generate_sct_detector_hexes(cfg)          # (N_det, 8, 3)
    print(f"  Detector crystals: {det_hex.shape[0]}")

    print("Generating SCT collimator plate hexahedron ...")
    plate_hex = generate_sct_collimator_hex(cfg)        # (1, 8, 3)

    print(f"Placing {cfg['num_holes']} collimator holes (seed={cfg['hole_seed']}) ...")
    holes = generate_sct_collimator_holes(cfg, seed=cfg["hole_seed"])
    print(f"  Holes placed: {holes.shape[0]}")

    print("Generating octagonal prism descriptors for hole voids ...")
    hole_prisms = generate_sct_hole_prisms(holes)
    print(f"  Prisms generated: {hole_prisms['centers'].shape[0]}")

    # FOV box for visualisation — driven by cfg so it always matches the config
    fov_hex = box_to_hex(
        0.0, 0.0, 0.0,
        cfg["fov_size_x_mm"],
        cfg["fov_size_y_mm"],
        cfg["fov_size_z_mm"],
    ).unsqueeze(0)

    # -----------------------------------------------------------------------
    # 4) Save .tensor file
    # -----------------------------------------------------------------------
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(_script_dir, "..", "..", "..", "data", "scanner_layouts")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "sct_single_position_base_3d.tensor")

    scanner_md5 = generate_md5_from_tensors(det_hex, plate_hex)
    torch.save(
        {
            "scanner MD5": scanner_md5,
            "applied_config": cfg,
            # (M, 5): [x_center, y_front, y_back, z_center, radius_mm]
            "collimator holes": holes,
            # Octagonal prism descriptors (Option A hole mechanism).
            # pymatcal loads these as OBJECT_TYPE_CONVEX_POLY with mu = -mu_plate
            # so that hole transmission cancels the solid-plate attenuation.
            "collimator hole prisms": hole_prisms,
            "layouts": {
                "position 000": {
                    "position": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                    "detector units 3d": det_hex,    # (N_det, 8, 3)
                    "plate segments 3d": plate_hex,  # (1, 8, 3) — full plate, no holes
                }
            },
        },
        out_file,
    )
    print(f"\nSaved 3D layout to:\n  {out_file}")
    print(f"  scanner MD5 : {scanner_md5}")

    # -----------------------------------------------------------------------
    # 5) 3D visualisation
    # -----------------------------------------------------------------------
    print("\nPlotting 3D scanner layout ...")
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Collimator plate
    plot_hexahedra_mpl3d(
        plate_hex, ax, facecolor="dimgray", edgecolor="black", alpha=0.4, wireframe=False
    )

    # Detector crystals — colour by layer for clarity
    layer_colours = ["steelblue", "dodgerblue", "deepskyblue", "gold"]
    offset = 0
    gap_y = cfg["detector_layer_gap_mm"]
    current_y = cfg["detector_y_start_mm"]
    for li, layer in enumerate(cfg["detector_layers"]):
        n = layer["nx"] * layer["nz"]
        layer_hexes = det_hex[offset : offset + n]
        plot_hexahedra_mpl3d(
            layer_hexes,
            ax,
            facecolor=layer_colours[li % len(layer_colours)],
            edgecolor="navy",
            alpha=0.5,
            wireframe=False,
        )
        offset += n
        current_y += layer["crystal_y_mm"] + gap_y

    # FOV box
    plot_hexahedra_mpl3d(
        fov_hex, ax, facecolor="magenta", edgecolor="magenta", alpha=0.07, wireframe=False
    )

    # Hole centres as scatter (first 200 for performance)
    n_show = min(200, holes.shape[0])
    hx_pts = holes[:n_show, 0].numpy()
    hy_pts = holes[:n_show, 1].numpy()
    hz_pts = holes[:n_show, 3].numpy()
    ax.scatter(hx_pts, hy_pts, hz_pts, c="white", s=2, alpha=0.6, zorder=5)

    autoscale_3d_to_points(ax, det_hex, plate_hex, margin=0.06)
    ax.set_title("SCT Flat-Panel Scanner Base Layout (3D)")

    plot_filename = "sct_base_layout_3d.png"
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    print(f"3D plot saved to {plot_filename}")
    plt.show()
