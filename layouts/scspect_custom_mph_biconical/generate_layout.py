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

if __name__ == "__main__":
    # --- 1. Define Configuration Parameters ---
    cfg = {
        # Active Detector Ring specs (4 concentric rings)
        "detector_width_mm": 0.84,
        "detector_thickness_mm": 6.0,
        "det_rings_r_in": [130.0, 195.0, 260.0, 325.0],
        "det_rings_n_crystals": [480, 720, 960, 1200],
        
        # Bi-conical Collimator specs (Double-tapered)
        "pinhole_diameter_mm": 0.4,
        "pinhole_opening_angle_deg": 27.0,
        "n_pinholes": 72,
        # Centered at 65mm radius, with a 20mm thickness, it spans 55mm to 75mm 
        # fitting safely inside the innermost 130mm detector ring.
        "collimator_ring_radius_mm": 65.0, 
        "collimator_thickness_mm": 2.5,
        
        "fov_radius": 10.0
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
    print(f"Total Solid Collimator Segments generated: {plate_segments.shape[0]}")

    # --- 3. Define the Single Base Layout ---
    base_layout = {
        "position": torch.tensor([0.0, 0.0, 0.0]),
        "detector units": detector_units,
        "plate segments": plate_segments,
    }
    
    output_data = {
        "scanner MD5": f"sc_spect_hybrid_mph_biconical_ççç",
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
    print("\nGenerating plot of the base scanner layout...")
    fig, ax = plt.subplots(figsize=(14, 14))
    
    plot_polygons_from_vertices_2d_mpl(
        base_layout["detector units"], ax, facecolor='orange', edgecolor='black', linewidth=0.2, label="Detectors (4 Rings)"
    )
    plot_polygons_from_vertices_2d_mpl(
        base_layout["plate segments"], ax, facecolor='gray', edgecolor='black', label="Bi-conical Collimator"
    )

    # FOV OVERLAY 
    plot_fov_circle(ax, cfg["fov_radius"], edgecolor="red", linewidth=2, linestyle="--")

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X (mm)", fontsize=18)
    ax.set_ylabel("Y (mm)", fontsize=18)
    ax.grid(True)
    ax.legend(fontsize=14)
    
    # Zoomed out view to see all 4 detector rings
    lim = 70 
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plot_filename = f"hybrid_sc_spect_mph_base_layout_{cfg['n_pinholes']}_{cfg['collimator_thickness_mm']}_{cfg['pinhole_diameter_mm']}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved to {plot_filename}")