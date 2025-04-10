if __name__ == "__main__":
    import os

    # check if the csv file exists

    import torch
    import matplotlib.pyplot as plt
    import pydetgen
    import pandas as pd
    # cvs_path = get_cvs_path(sys.argv)
    cvs_path = pydetgen.get_valid_cvs_path("tmp/panel.csv")
    
    # Read the CSV file
    df = pd.read_csv(cvs_path)
    # Get the polygon groups from the CSV file
    
    
    plate_ploygons_tensor = torch.cat(
        list(pydetgen.get_polygon_groups_from_csv(["plate_0"], cvs_path))
    )
    crystal_polygon_tensor = torch.cat(
        list(pydetgen.get_polygon_groups_from_csv(["crystals_0"], cvs_path))
    )
    
    
    plate_polygon_center_tensor = torch.mean(plate_ploygons_tensor, dim=1)
    crystal_polygon_center_tensor = torch.mean(crystal_polygon_tensor, dim=1)

    aperture_center_tensor = pydetgen.get_plate_aperture_center(
        plate_ploygons_tensor
    )


    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 10), layout="constrained")
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

    ax.plot(
        aperture_center_tensor[:, 0],
        aperture_center_tensor[:, 1],
        "ro",
        markersize=4,
        label="Aperture Center",
    )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.autoscale()

    for i, center in enumerate(plate_polygon_center_tensor):
        ax.text(
            x=center[0].item(),
            y=center[1].item(),
            ha="center",
            va="center",
            s=str(i),
            fontsize=8,
            color="black",
        )

    for i, center in enumerate(crystal_polygon_center_tensor):
        ax.text(
            x=center[0].item(),
            y=center[1].item(),
            ha="center",
            va="center",
            s=str(i),
            fontsize=8,
            color="black",
        )

    # Create output directory if it doesn't exist
    out_dir = "output/plots"
    os.makedirs(out_dir, exist_ok=True)

    # Save the plot
    out_file_path = os.path.join(out_dir, f"{'panel_0_geometry'}.png")
    plt.savefig(out_file_path, dpi=150)
    print(f"Plot saved to {out_file_path}")
