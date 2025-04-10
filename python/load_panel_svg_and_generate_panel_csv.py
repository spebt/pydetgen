if __name__ == "__main__":
    import torch
    import os
    import pydetgen
    import matplotlib.pyplot as plt

    # from matplotlib.collections import PolyCollection

    import pandas as pd

    try:
        plt.close()
    except Exception as e:
        pass

    group_names = ["panel_0", "plate_0"]

    # SVG file path
    svg_path = "panel.svg"

    scanner_geometry = pydetgen.get_scanner_geometry_from_svg(
        svg_path, group_names
    )

    plate_polygon_0_tensor = scanner_geometry["plate_0"]
    crystal_polygon_0_tensor = scanner_geometry["panel_0"]

    plate_patch_polygon = torch.tensor(
        [
            [127.6756, 93.1407],
            [126.6756, 91.4086],
            [142.5000, 64.0000],
            [144.5000, 64.0000],
        ]
    )

    plate_polygon_0_tensor = torch.cat(
        [plate_polygon_0_tensor, plate_patch_polygon.unsqueeze(0)]
    )

    # Sort the plate polygon by the center xy coordinates
    plate_polygon_0_centers = torch.mean(plate_polygon_0_tensor, dim=1)
    plate_polygon_0_centers_indices = (
        pydetgen.get_polygon_vertices_sorted_by_y_batch(
            plate_polygon_0_centers.unsqueeze(0)
        )
    )[1][0]

    crystal_polygon_0_centers = torch.mean(crystal_polygon_0_tensor, dim=1)
    # Sort the crystal polygon by the center xy coordinates
    crystal_polygon_0_centers_indices = (
        pydetgen.get_polygon_vertices_sorted_by_xy_batch(
            crystal_polygon_0_centers.unsqueeze(0)
        )
    )[1][0]

    plate_polygon_0_tensor = plate_polygon_0_tensor[
        plate_polygon_0_centers_indices
    ]
    crystal_polygon_0_tensor = crystal_polygon_0_tensor[
        crystal_polygon_0_centers_indices
    ]

    plate_polygon_0_tensor = pydetgen.get_polygon_vertices_sorted_by_xy_batch(
        plate_polygon_0_tensor
    )[0]

    crystal_polygon_0_tensor = pydetgen.get_polygon_vertices_sorted_by_xy_batch(
        crystal_polygon_0_tensor
    )[0]

    # sort the vertices of the plate polygon by the angle
    plate_polygon_0_tensor = pydetgen.get_polygon_vertices_sorted_by_rad_batch(
        plate_polygon_0_tensor
    )
    # sort the vertices of the crystal polygon by the angle
    crystal_polygon_0_tensor = (
        pydetgen.get_polygon_vertices_sorted_by_rad_batch(
            crystal_polygon_0_tensor
        )
    )

    colums = ["polygon group", "polygon id", "vertice id", "x", "y"]
    scanner_vertices_list = []
    for i, polygon in enumerate(plate_polygon_0_tensor):
        for j, vert in enumerate(polygon):
            scanner_vertices_list.append(
                ["plate_0", i, j, vert[0].item(), vert[1].item()]
            )
    for i, polygon in enumerate(crystal_polygon_0_tensor):
        for j, vert in enumerate(polygon):
            scanner_vertices_list.append(
                ["crystals_0", i, j, vert[0].item(), vert[1].item()]
            )
    df = pd.DataFrame(scanner_vertices_list, columns=colums)

    tmp_dir = "tmp"
    csv_path = tmp_dir + "/panel.csv"

    # Create the tmp directory if it does not exist
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    df.to_csv(csv_path)
    print(f"CSV file saved to {csv_path}")
