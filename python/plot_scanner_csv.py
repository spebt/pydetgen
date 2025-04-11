def get_cvs_path(args, script_name="plot_scanner_csv.py"):
    """
    Get the csv file path from the command line arguments.
    """
    import sys

    if len(args) < 2:
        print(f"Usage: python {script_name} <csv_file_path>")
        sys.exit(1)
    return args[1]


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    import pydetgen
    import os, sys

    script_name = os.path.basename(__file__)

    cvs_path = get_cvs_path(sys.argv, script_name=script_name)
    cvbs_path = pydetgen.get_valid_cvs_path(cvs_path)
    plate_groups = [f"plate_{i}" for i in range(0, 6)]
    crystal_groups = [f"crystals_{i}" for i in range(0, 6)]
    plate_ploygons_tensor = torch.cat(
        list(pydetgen.get_polygon_groups_from_csv(plate_groups, cvs_path))
    )
    crystal_polygon_tensor = torch.cat(
        list(pydetgen.get_polygon_groups_from_csv(crystal_groups, cvs_path))
    )

    # Set up the plot
    fig, ax = plt.subplots(figsize=(20, 20), layout="constrained")
    ax.set_title("Panel Geometry")
    ax.set_aspect("equal")
    pydetgen.plot_polygons_from_vertices_mpl(
        plate_ploygons_tensor, ax, color="blue", alpha=0.5
    )
    pydetgen.plot_polygons_from_vertices_mpl(
        crystal_polygon_tensor, ax, color="orange", alpha=0.5
    )
    pydetgen.plot_fov_as_rectangle_mpl(
        {"physical dimensions": [128, 128], "center coordinates": [0, 0]},
        ax,
        fc="none",
        ec="green",
        alpha=0.5,
    )
    crystal_polygon_center_tensor = torch.mean(crystal_polygon_tensor, dim=1)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.autoscale()
    for i, center in enumerate(crystal_polygon_center_tensor):
        ax.text(
            x=center[0].item(),
            y=center[1].item(),
            ha="center",
            va="center",
            s=str(i),
            fontsize=6,
            color="black",
        )

    out_dir = "output/plots"
    # Create the output directory if it does not exist
    os.makedirs(out_dir, exist_ok=True)
    # Save the plot
    out_file_path = os.path.join(out_dir, f"{os.path.basename(cvs_path).split('.csv')[0]}_geometry.png")
    plt.savefig(out_file_path, dpi=150)
    print(f"Plot saved to {out_file_path}")
