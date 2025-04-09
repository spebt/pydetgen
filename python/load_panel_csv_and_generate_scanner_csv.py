import torch
import matplotlib.pyplot as plt
import pandas as pd
from polygon_transform_2d_helper import rotate_polygons
from rich.progress import Progress, BarColumn


def get_polygon_groups_from_csv(grp_names: list[str], csv_path: str):
    df = pd.read_csv(csv_path)
    for grp_name in grp_names:
        grp_verts = df.loc[df["polygon group"] == grp_name]
        grp_tensor = torch.empty(
            grp_verts.shape[0] // 4, 4, 2, dtype=torch.float
        )
        # print(grp_verts["polygon id"].unique())
        grp_tensor[
            torch.tensor(grp_verts["polygon id"].values, dtype=torch.int),
            torch.tensor(grp_verts["vertice id"].values, dtype=torch.int),
            :,
        ] = torch.tensor(grp_verts[["x", "y"]].values, dtype=torch.float)
        yield grp_tensor


def set_plate_aperture_width(
    width: float, plate_tensor: torch.Tensor
) -> torch.Tensor:
    aperture_plate_polygons = plate_tensor[:-1]
    aperture_centers = (
        (
            (
                aperture_plate_polygons[1:, 0, 1]
                + aperture_plate_polygons[:-1, -1, 1]
            )
            * 0.5
        )
        .unsqueeze(1)
        .expand(-1, 2)
    )
    aperture_plate_polygons[1:, :2, 1] = aperture_centers + width * 0.5
    aperture_plate_polygons[:-1, 2:, 1] = aperture_centers - width * 0.5
    return torch.cat([aperture_plate_polygons, plate_tensor[-1].unsqueeze(0)])


def get_df_rows_from_polygon_tensor(
    polygon_tensor: torch.Tensor, group_name: str
):
    for i, polygon in enumerate(polygon_tensor):
        for j, vert in enumerate(polygon):
            yield [group_name, i, j, vert[0].item(), vert[1].item()]


if __name__ == "__main__":
    import os, sys

    group_names = ["plate_0", "crystals_0"]

    tmp_dir = "tmp"
    csv_path = tmp_dir + "/scanner.csv"
    out_dir = "output"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Please run the script to generate the scanner.csv file first."
        )
        sys.exit(1)

    if not os.path.exists("output"):
        os.makedirs("output")
    plate_polygon_0_tensor, xtal_polygon_0_tensor = get_polygon_groups_from_csv(
        group_names, csv_path
    )

    aperture_w_list = torch.arange(1, 5, 0.5)
    with Progress(
        BarColumn(),
        "[progress.description]{task.description}",
        "{task.completed}/{task.total}",
        # "elapsed time: {elapsed}",
        # "remaining time: {remaining}",
    ) as progress:
        task1 = progress.add_task(
            "[cyan]Processing...", total=aperture_w_list.shape[0]
        )
        for aperture_w in aperture_w_list:
            plate_polygon_0_tensor = set_plate_aperture_width(
                aperture_w.item(), plate_polygon_0_tensor
            )
            plate_polygon_tensor = plate_polygon_0_tensor
            xtal_polygon_tensor = xtal_polygon_0_tensor

            colums = ["polygon group", "polygon id", "vertice id", "x", "y"]
            scanner_vertices_list = []
            for angle_id in torch.arange(6):
                angle_rad = angle_id * 60 / 180 * torch.pi
                plate_polygon_tensor_at_rotation = rotate_polygons(
                    plate_polygon_0_tensor, angle_rad
                )
                xtal_polygon_tensor_at_rotation = rotate_polygons(
                    xtal_polygon_0_tensor, angle_rad
                )
                scanner_vertices_list.extend(
                    get_df_rows_from_polygon_tensor(
                        plate_polygon_tensor_at_rotation,
                        "plate_{}".format(angle_id),
                    )
                )
                scanner_vertices_list.extend(
                    get_df_rows_from_polygon_tensor(
                        xtal_polygon_tensor_at_rotation,
                        "crystals_{}".format(angle_id),
                    )
                )
                plate_polygon_tensor = torch.cat(
                    [
                        plate_polygon_tensor,
                        plate_polygon_tensor_at_rotation,
                    ]
                )
                xtal_polygon_tensor = torch.cat(
                    [
                        xtal_polygon_tensor,
                        xtal_polygon_tensor_at_rotation,
                    ]
                )

            df = pd.DataFrame(scanner_vertices_list, columns=colums)

            df.to_csv(
                out_dir + "/scanner_{}_mm_aperture.csv".format(aperture_w),
                index=False,
            )
            progress.update(task1, advance=1)
        progress.refresh()
