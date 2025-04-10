import torch
import matplotlib.pyplot as plt
import pandas as pd
import pydetgen

from rich.progress import Progress, BarColumn


if __name__ == "__main__":
    import os, sys

    group_names = ["plate_0", "crystals_0"]

    tmp_dir = "tmp"
    csv_path = tmp_dir + "/panel.csv"

    if not os.path.exists(csv_path):
        prerequisite_script = "load_panel_svg_and_generate_panel_csv.py"
        raise FileNotFoundError(
            f"Please use {prerequisite_script} to generate the csv file."
        )

    out_dir = "output/scanner_csv"
    os.makedirs(out_dir, exist_ok=True)

    plate_polygon_0_tensor, xtal_polygon_0_tensor = (
        pydetgen.get_polygon_groups_from_csv(group_names, csv_path)
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
            plate_polygon_0_tensor = pydetgen.set_plate_aperture_width(
                aperture_w.item(), plate_polygon_0_tensor
            )
            plate_polygon_tensor = plate_polygon_0_tensor
            xtal_polygon_tensor = xtal_polygon_0_tensor

            colums = ["polygon group", "polygon id", "vertice id", "x", "y"]
            scanner_vertices_list = []
            for angle_id in torch.arange(6):
                angle_rad = angle_id * 60 / 180 * torch.pi
                plate_polygon_tensor_at_rotation = pydetgen.rotate_polygons(
                    plate_polygon_0_tensor, angle_rad
                )
                xtal_polygon_tensor_at_rotation = pydetgen.rotate_polygons(
                    xtal_polygon_0_tensor, angle_rad
                )
                scanner_vertices_list.extend(
                    pydetgen.get_df_rows_from_polygon_tensor(
                        plate_polygon_tensor_at_rotation,
                        "plate_{}".format(angle_id),
                    )
                )
                scanner_vertices_list.extend(
                    pydetgen.get_df_rows_from_polygon_tensor(
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
            out_file_path = out_dir + "/scanner_{}_mm_aperture.csv".format(
                aperture_w
            )
            df.to_csv(
                out_file_path,
                index=False,
            )
            print(f"Saved to {out_file_path}")
            progress.update(task1, advance=1)
        progress.refresh()
