import torch
import os
import matplotlib.pyplot as plt

from helper import (
    generate_sc_spect_detectors,
    generate_biconical_collimators,
    plot_polygons_from_vertices_2d_mpl,
)

def plot_fov_circle(ax, radius_mm: float, center=(0.0, 0.0), **kwargs):
    """Plot a circular FOV overlay."""
    cx, cy = center
    circle = plt.Circle((cx, cy), radius_mm, fill=False, **kwargs)
    ax.add_patch(circle)


def plot_fov_square(ax, side_length_mm: float, center=(0.0, 0.0), **kwargs):
    """Plot a square FOV centered on the scanner axis."""
    cx, cy = center
    half_side = side_length_mm / 2.0
    square = plt.Rectangle(
        (cx - half_side, cy - half_side),
        side_length_mm,
        side_length_mm,
        fill=False,
        **kwargs,
    )
    ax.add_patch(square)


def save_layout_plot(base_layout, cfg, output_path, plot_limit, closeup=False):
    """Save either the complete scanner or a collimator-ring close-up."""
    fig, ax = plt.subplots(figsize=(12, 12))

    if not closeup:
        plot_polygons_from_vertices_2d_mpl(
            base_layout["detector units"],
            ax,
            facecolor="lightblue",
            edgecolor="blue",
            linewidth=0.18,
            label="Detector crystals (4 rings)",
        )
    plot_polygons_from_vertices_2d_mpl(
        base_layout["plate segments"],
        ax,
        facecolor="gray",
        edgecolor="black",
        linewidth=0.65 if closeup else 0.35,
        label="Biconical collimator ring",
    )
    plot_fov_square(
        ax,
        cfg["fov_side_length_mm"],
        edgecolor="red",
        linewidth=2.2,
        linestyle=":",
        label="FOV (10 mm x 10 mm)",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)", fontsize=18)
    ax.set_ylabel("Y (mm)", fontsize=18)
    ax.grid(True, alpha=0.12)
    ax.legend(fontsize=14, loc="upper right")
    ax.set_xlim([-plot_limit, plot_limit])
    ax.set_ylim([-plot_limit, plot_limit])
    ax.tick_params(axis="both", which="major", labelsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    # --- 1. Define Configuration Parameters ---
    cfg = {
        # Active Detector Ring specs (4 concentric rings)
        "detector_width_mm": 0.84,
        "detector_thickness_mm": 6.0,
        "detector_axial_length_mm": 20.0,
        "detector_tangential_gap_mm": 0.84,
        # Forty 9-degree cassette positions; each cassette has one block per ring.
        "n_cassette_positions": 40,
        "det_rings_r_in": [130.0, 195.0, 260.0, 325.0],
        "det_rings_n_crystals": [480, 720, 960, 1200],
        
        # Bi-conical Collimator specs (Double-tapered)
        "pinhole_diameter_mm": 0.4,
        "pinhole_opening_angle_deg": 27.0,
        "n_pinholes": 72,
        # Centered at 65 mm radius with 2.5 mm total radial thickness.
        "collimator_ring_radius_mm": 65.0, 
        "collimator_thickness_mm": 2.5,
        
        "fov_side_length_mm": 10.0,
    }

    # --- 2. Generate the Hybrid Geometry ---
    print("Generating 4-ring SC-SPECT active detectors...")
    detector_units = generate_sc_spect_detectors(cfg)
    
    print("Generating MPH bi-conical collimators...")
    inner_collimator, outer_collimator = generate_biconical_collimators(
        pinhole_diameter_mm=cfg["pinhole_diameter_mm"],
        pinhole_opening_angle_deg=cfg["pinhole_opening_angle_deg"],
        n_pinholes=cfg["n_pinholes"],
        collimator_ring_radius_mm=cfg["collimator_ring_radius_mm"],
        collimator_thickness_mm=cfg["collimator_thickness_mm"]
    )
    
    # Combine inner and outer tapered parts into a single tensor
    plate_segments = torch.cat((inner_collimator, outer_collimator), dim=0)

    print(f"Total Detector Crystals generated: {detector_units.shape[0]}")
    print(
        f"Detector cassette positions generated: {cfg['n_cassette_positions']} "
        "(one detector block per ring at each position)"
    )
    print(f"Total Solid Collimator Segments generated: {plate_segments.shape[0]}")

    # --- 3. Define the Single Base Layout ---
    base_layout = {
        "position": torch.tensor([0.0, 0.0, 0.0]),
        "detector units": detector_units,
        "plate segments": plate_segments,
    }
    
    output_data = {
        "scanner MD5": f"sc_spect_hybrid_mph_biconical_ççç",
        "applied_config": dict(cfg),
        "motion_parameters": {
            "n_rotational_steps_defined": 1,
            "n_translational_shifts_grid": [1, 1],
            "translational_step_size_mm": [0.0, 0.0],
            "generated_n_positions": 1
        },
        "layouts": {f"position {0:03d}": base_layout}
    }

    # --- 4. Save the Layout ---
    output_dir = "/vscratch/grp-rutaoyao/sid/data/scanner_layouts/"
    os.makedirs(output_dir, exist_ok=True)
    out_file_name = os.path.join(output_dir, f"hybrid_sc_spect_mph_biconical_base_{cfg['n_pinholes']}_{cfg['collimator_thickness_mm']}_{cfg['pinhole_diameter_mm']}.tensor")
    
    print(f"\nSaving hybrid layout to:\n  {out_file_name}")
    torch.save(output_data, out_file_name)
    print("Layout saved successfully.")

    # --- 5. VISUALIZATION ---
    manuscript_dir = "../../../documents/SPIE/Manuscript"
    full_plot_filename = os.path.join(
        manuscript_dir, "SC_SPECT_full_configuration.png"
    )
    closeup_plot_filename = os.path.join(
        manuscript_dir, "SC_SPECT_collimator_ring_closeup.png"
    )

    print("\nGenerating full scanner configuration plot...")
    save_layout_plot(base_layout, cfg, full_plot_filename, plot_limit=350.0)
    print(f"Plot saved to {full_plot_filename}")

    print("Generating collimator-ring close-up plot...")
    save_layout_plot(
        base_layout,
        cfg,
        closeup_plot_filename,
        plot_limit=72.0,
        closeup=True,
    )
    print(f"Plot saved to {closeup_plot_filename}")
