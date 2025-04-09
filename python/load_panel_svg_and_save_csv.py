if __name__ == "__main__":
    import torch
    import os
    from scanner_svg_helper import (
        get_scanner_geometry_from_svg,
        get_verts_sorted_by_xy_2d,
    )
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    import pandas as pd

    try:
        plt.close()
    except Exception as e:
        pass

    group_names = ["panel_0", "plate_0"]
    scanner_geometry = get_scanner_geometry_from_svg("scanner.svg", group_names)

    plate_polygon_0_tensor = scanner_geometry["plate_0"]
    xtal_polygon_0_tensor = scanner_geometry["panel_0"]

    plate_patch_polygon = torch.tensor(
        [
            [127.6756, 93.1407],
            [126.6756, 91.4086],
            [142.5000, 64.0000],
            [144.5000, 64.0000],
        ]
    )

    plate_polygon_0_tensor = torch.cat(
        [plate_patch_polygon.unsqueeze(0), plate_polygon_0_tensor]
    )

    # Sort the plate polygon by the center xy coordinates
    plate_polygon_0_centers = torch.mean(plate_polygon_0_tensor, dim=1)
    _, polygon_indices = get_verts_sorted_by_xy_2d(plate_polygon_0_centers)
    plate_polygon_0_tensor = plate_polygon_0_tensor[polygon_indices]

    # sort each polygon vertices by xy
    plate_polygon_0_tensor = torch.stack(
        list(
            map(
                lambda polygon: get_verts_sorted_by_xy_2d(polygon)[0],
                plate_polygon_0_tensor,
            )
        )
    )
    plate_polygon_0_tensor = plate_polygon_0_tensor[:, [0, 1, 3, 2], :]

    colums = ["polygon group", "polygon id", "vertice id", "x", "y"]
    scanner_vertices_list = []
    for i, polygon in enumerate(plate_polygon_0_tensor):
        for j, vert in enumerate(polygon):
            scanner_vertices_list.append(
                ["plate_0", i, j, vert[0].item(), vert[1].item()]
            )
    for i, polygon in enumerate(xtal_polygon_0_tensor):
        for j, vert in enumerate(polygon):
            scanner_vertices_list.append(
                ["crystals_0", i, j, vert[0].item(), vert[1].item()]
            )
    df = pd.DataFrame(scanner_vertices_list, columns=colums)

    tmp_dir = "tmp"
    csv_path = tmp_dir + "/scanner.csv"

    # Create the tmp directory if it does not exist
    if not os.path.exists(tmp_dir):   
        os.makedirs(tmp_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
