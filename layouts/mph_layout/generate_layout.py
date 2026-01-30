# main_script.py
import torch
from torch import save as torch_save, cat, tensor
import os
import matplotlib.pyplot as plt
import numpy as np

# Assuming helper.py is in the same directory
# Make sure helper.py is available in the same directory as this script.
from helper import (
    generate_transaxial_spect_geometry,
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
        "pinhole_diameter_mm": 3.0,
        "pinhole_opening_angle_deg": 27.0,
        "n_pinholes": 18,
        "collimator_ring_radius_mm": 215.0,
        "collimator_thickness_mm": 20.0,
        "detector_ring_radius_mm": 215.0 + 542.0,
        "detector_thickness_mm": 6,
        "detector_crystal_width_mm": 3.5,
    }

    # --- 2. Generate the Base Scanner Layout ---
    print("Generating bi-conical (double-tapered) pinhole geometry...")
    detector_units, inner_collimator, outer_collimator = generate_transaxial_spect_geometry(**cfg)

    # --- 3. Combine Collimator Parts ---
    combined_plate_segments = cat((inner_collimator, outer_collimator), dim=0)
    print(f"\nCombined inner and outer collimator rings into a single tensor.")

    # --- 4. Define the Single Base Layout ---
    # Rotation logic has been removed. We are only defining the base layout.
    base_layout = {
        "position": tensor([0.0, 0.0, 0.0]),
        "detector units": detector_units,
        "plate segments": combined_plate_segments,
    }
    
    # The 'all_layouts' dictionary now contains only the single base layout.
    output_data = {
        "scanner MD5": "placeholder_md5_for_biconical_single_position_scanner",
        "motion_parameters": {
            "n_rotational_steps_defined": 1,
            "n_translational_shifts_grid": [1, 1],
            "translational_step_size_mm": [0.0, 0.0],
            "generated_n_positions": 1
        },
        "layouts": {f"position {0:03d}": base_layout}
    }

    # --- 5. Save the Layout to a Single .tensor File ---
    output_dir = "../data/scanner_layouts/"
    os.makedirs(output_dir, exist_ok=True)
    out_file_name = os.path.join(output_dir, "mph_hourglass_single_position_base_2mm_18pinholes.tensor")
    
    print(f"\nSaving single SPECT layout to:\n  {out_file_name}")
    torch_save(output_data, out_file_name)
    print("\nLayout saved successfully.")

    # --- 6. VISUALIZATION SECTION ---
    print("\nGenerating plot of the base scanner layout...")

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot the detector and collimator components from the base layout
    plot_polygons_from_vertices_2d_mpl(
        base_layout["detector units"], ax, facecolor='lightblue', edgecolor='blue', label="Detectors"
    )
    plot_polygons_from_vertices_2d_mpl(
        base_layout["plate segments"], ax, facecolor='gray', edgecolor='black', label="Collimator"
    )

    # --- FOV OVERLAY (CIRCLE) ---
    fov_radius_mm = 64.0  # example; set this to your actual FOV radius
    plot_fov_circle(ax, fov_radius_mm, edgecolor="red", linewidth=2, linestyle="--")
    # ax.text(0, fov_radius_mm + 10, f"FOV radius = {fov_radius_mm:.1f} mm",
            # ha="center", va="bottom", fontsize=12)

    # Finalize and show the plot
    ax.set_aspect('equal', adjustable='box')
    # ax.set_title("SPECT Scanner Base Layout")
    ax.set_xlabel("X (mm)", fontsize=18)
    ax.set_ylabel("Y (mm)", fontsize=18)
    ax.grid(True)
    ax.legend(fontsize=14)
    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-1000, 1000])
    ax.tick_params(axis='both', which='major', labelsize=14)
    plot_filename = "spect_base_layout.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()