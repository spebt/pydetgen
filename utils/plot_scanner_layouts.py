import matplotlib.pyplot as plt
import matplotlib.patches as patches  # <-- NEW: Required for drawing the FOV rectangle
from matplotlib.collections import PolyCollection
import sys
import os
from torch import (
    load as torch_load,
    empty as empty_tensor,
    Tensor,
    pi
)
from matplotlib.axes import Axes

def plot_polygons_from_vertices_2d_mpl(vertices: Tensor, ax: Axes, **kwargs):
    """Plots a collection of polygons on a matplotlib Axes object."""
    p = PolyCollection(vertices.tolist(), **kwargs)
    ax.add_collection(p)
    return p

if __name__ == "__main__":

    # --- FOV Configuration ---
    # Adjust these variables to match your reconstruction parameters
    FOV_PIXELS_X, FOV_PIXELS_Y = 512, 512
    MM_PER_PIXEL_X, MM_PER_PIXEL_Y = 0.25, 0.25
    FOV_CENTER_X, FOV_CENTER_Y = 0.0, 0.0

    # Calculate the total physical size of the FOV in mm
    fov_size_x = FOV_PIXELS_X * MM_PER_PIXEL_X
    fov_size_y = FOV_PIXELS_Y * MM_PER_PIXEL_Y
    # -------------------------

    # Load the scanner layouts
    filename = sys.argv[1]

    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        raise FileNotFoundError(f"File {filename} does not exist.")

    filename_unique_id = filename.split(".")[0].split("_")[-1]
    scanner_layouts_data = torch_load(filename, weights_only=False)["layouts"]

    try:
        position_indices = list(map(int, sys.argv[2].split(",")))
    except Exception as e:
        print(
            "Usage: python plot_scanner_layout.py <filename> <layout_position_indices_comma_separated>"
        )
        sys.exit(1)

    # Plot the detector units
    fig, ax = plt.subplots(layout="constrained", figsize=(10, 10))
    ax.set_title(f"Scanner layout {position_indices[0]:03d}")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_aspect("equal")
    
    # You may want to adjust these limits if your FOV is larger than 256x256 mm
    ax.set_xlim(-512, 512)
    ax.set_ylim(-512, 512)

    # Plot dummy detector units for legend
    detector_units_polygons_collection = plot_polygons_from_vertices_2d_mpl(
        empty_tensor((1, 4, 2)),
        ax=ax,
        fc="C0",
        ec="none",
    )

    # Plot dummy plate segments for legend
    plate_segments_polygons_collection = plot_polygons_from_vertices_2d_mpl(
        empty_tensor((1, 4, 2)),
        ax=ax,
        fc="C1",
        ec="none",
    )

    # --- Draw the Red Dotted FOV ---
    # Calculate the lower-left corner coordinate for the rectangle patch
    bottom_left_x = FOV_CENTER_X - (fov_size_x / 2)
    bottom_left_y = FOV_CENTER_Y - (fov_size_y / 2)
    
    fov_patch = patches.Rectangle(
        (bottom_left_x, bottom_left_y),
        fov_size_x,
        fov_size_y,
        linewidth=1.5,
        edgecolor='red',
        facecolor='none',
        linestyle=':'  # ':' creates a dotted line. Use '--' if you prefer dashed.
    )
    ax.add_patch(fov_patch)

    # Add a legend that includes the FOV
    ax.legend(
        [
            detector_units_polygons_collection,
            plate_segments_polygons_collection,
            fov_patch
        ],
        [
            "Detector units", 
            "Plate segments", 
            f"FOV ({fov_size_x} x {fov_size_y} mm)"
        ],
        loc="upper right",
    )
    
    for i in position_indices:
        # Extract the detector units and plate segments
        detector_units = scanner_layouts_data[f"position {i:03d}"][
            "detector units"
        ]
        plate_segments = scanner_layouts_data[f"position {i:03d}"][
            "plate segments"
        ]
        position = scanner_layouts_data[f"position {i:03d}"][
            "position"
        ].tolist()
        angle = float(position[0] * 180 / pi)

        print(
            f"Plotting scanner layout {i:03d} with {detector_units.shape[0]} detector units and {plate_segments.shape[0]} plate segments"
        )

        # Remove the previous detector units and plate segments
        if "detector_units_polygons_collection" in locals():
            detector_units_polygons_collection.remove()
        if "plate_segments_polygons_collection" in locals():
            plate_segments_polygons_collection.remove()

        ax.set_title(
            f"Scanner Position ID: {i:03d}, Transformation: (rotation: {angle:.2f}°, x: {position[1]:.2f} mm, y: {position[2]:.2f} mm)"
        )

        # Plot the detector units
        detector_units_polygons_collection = plot_polygons_from_vertices_2d_mpl(
            detector_units,
            ax=ax,
            fc="C0",
            ec="none",
        )
        # Plot the plate segments
        plate_segments_polygons_collection = plot_polygons_from_vertices_2d_mpl(
            plate_segments,
            ax=ax,
            fc="C1",
            ec="none",
        )

        fig.savefig(
            f"scanner_layout_{i:03d}_{filename_unique_id}.png",
            dpi=150,
            bbox_inches="tight",
        )