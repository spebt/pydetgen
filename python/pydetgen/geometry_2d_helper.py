import torch
from matplotlib.axes import Axes
from matplotlib.collections import (
    PolyCollection,
)
import pandas as pd


def rotate_polygons(polygons, angle):
    verts_tensor = polygons.view(-1, 2)
    rot_mat = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)],
        ]
    )
    return (
        torch.bmm(
            rot_mat.unsqueeze(0).expand(verts_tensor.shape[0], -1, -1),
            verts_tensor.unsqueeze(-1),
        )
        .squeeze(-1)
        .view(polygons.shape)
    )


def get_polygon_vertices_sorted_by_y_batch(
    vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sorts the vertices of polygons in a batch by their x and y coordinates.
    Arguments:
        vertices (torch.Tensor): A tensor of shape (batch_size, num_vertices, 2)
            representing the vertices of the polygons.
    Returns:
        tuple: A tuple containing:
            - sorted_vertices (torch.Tensor): A tensor of shape (batch_size, num_vertices, 2)
                representing the sorted vertices.
            - indices (torch.Tensor): A tensor of shape (batch_size, num_vertices)
                representing the indices used for sorting.
    """

    # Get unique y values
    unique_y, unique_y_index = torch.unique(
        vertices[:, :, 1], return_inverse=True, sorted=True
    )
    # Sort the indices based on x and y values
    indices_by_y = torch.argsort(unique_y_index)
    return (
        vertices.gather(1, indices_by_y.unsqueeze(-1).expand(-1, -1, 2)),
        indices_by_y,
    )


def get_polygon_vertices_sorted_by_xy_batch(
    vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sorts the vertices of polygons in a batch by their x and y coordinates.
    Arguments:
        vertices (torch.Tensor): A tensor of shape (batch_size, num_vertices, 2)
            representing the vertices of the polygons.
    Returns:
        tuple: A tuple containing:
            - sorted_vertices (torch.Tensor): A tensor of shape (batch_size, num_vertices, 2)
                representing the sorted vertices.
            - indices (torch.Tensor): A tensor of shape (batch_size, num_vertices)
                representing the indices used for sorting.
    """

    # Get unique x values
    unique_x, unique_x_index = torch.unique(
        vertices[:, :, 0], return_inverse=True, sorted=True
    )
    # Get unique y values
    unique_y, unique_y_index = torch.unique(
        vertices[:, :, 1], return_inverse=True, sorted=True
    )
    # Sort the indices based on x and y values
    indices_by_xy = torch.argsort(unique_x_index * 100 + unique_y_index)
    return (
        vertices.gather(1, indices_by_xy.unsqueeze(-1).expand(-1, -1, 2)),
        indices_by_xy,
    )


def get_vertices_sorted_by_rad_raw_2d_batch(
    vertices: torch.Tensor, ref_point: torch.Tensor
) -> torch.Tensor:
    # sort the vertices by angle to point ref_point

    n_points = vertices.shape[1]
    rads = torch.atan2(
        vertices[:, :, 1] - ref_point[:, :, 1],
        vertices[:, :, 0] - ref_point[:, :, 0],
    )
    order = torch.argsort(rads, dim=1)
    return vertices.gather(dim=1, index=order.unsqueeze(-1).expand(-1, -1, 2))


def get_vertices_sorted_by_rad_2d_batch(
    vertices: torch.Tensor, ref_point: torch.Tensor
) -> torch.Tensor:
    # sort the vertices by angle to point ref_point

    n_points = vertices.shape[1]
    rads = torch.atan2(
        vertices[:, :, 1] - ref_point[:, :, 1],
        vertices[:, :, 0] - ref_point[:, :, 0],
    )
    rads = torch.where(rads < 0, rads + 2 * torch.pi, rads)
    order = torch.argsort(rads, dim=1)
    return vertices.gather(dim=1, index=order.unsqueeze(-1).expand(-1, -1, 2))


def get_polygon_vertices_sorted_by_rad_batch(vertices: torch.Tensor):
    """
    Sorts the vertices of polygons in a batch by their angle to the
    first vertice of each plolygon.
    Arguments:
        vertices (torch.Tensor): A tensor of shape (batch_size, num_vertices, 2)
            representing the vertices of the polygons.
    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_vertices, 2)
            representing the sorted vertices.
    """

    return torch.cat(
        [
            vertices[:, :1, :],
            get_vertices_sorted_by_rad_raw_2d_batch(
                vertices[:, 1:, :],
                vertices[:, :1, :].expand(-1, vertices[:, 1:, :].shape[1], -1),
            ),
        ],
        dim=1,
    )


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


def get_plate_aperture_center(plate_tensor: torch.Tensor) -> torch.Tensor:

    aperture_plate_polygons = plate_tensor[:-1]
    return (
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


def get_df_rows_from_polygon_tensor(
    polygon_tensor: torch.Tensor, group_name: str
):
    for i, polygon in enumerate(polygon_tensor):
        for j, vert in enumerate(polygon):
            yield [group_name, i, j, vert[0].item(), vert[1].item()]


# Plotting functions


def plot_polygons_from_vertices_mpl(vertices: torch.Tensor, ax: Axes, **kwargs):
    p = PolyCollection(vertices.tolist(), **kwargs)
    ax.add_collection(p)
    return p


def get_fov_corners(fov_dict: dict):
    fov_dims = torch.tensor(fov_dict["physical dimensions"])
    fov_corners = torch.tensor(
        [
            [-fov_dims[0] / 2, -fov_dims[1] / 2],
            [fov_dims[0] / 2, -fov_dims[1] / 2],
            [fov_dims[0] / 2, fov_dims[1] / 2],
            [-fov_dims[0] / 2, fov_dims[1] / 2],
        ]
    ) + torch.tensor(fov_dict["center coordinates"])
    return fov_corners


def plot_fov_as_rectangle_mpl(fov_dict: dict, ax: Axes, **kwargs):
    fov_corners = get_fov_corners(fov_dict)
    return plot_polygons_from_vertices_mpl(
        fov_corners.unsqueeze(0), ax, **kwargs
    )


def plot_scanner_from_vertices_2d_mpl(
    plate_polygon_tensor, xtal_polygon_tensor, ax, fov_dict
):

    plate_polycoll = plot_polygons_from_vertices_mpl(
        plate_polygon_tensor, ax, color="b", alpha=0.5
    )
    crystal_polycoll = plot_polygons_from_vertices_mpl(
        xtal_polygon_tensor, ax, color="orange", alpha=0.5
    )
    fov_polycoll = plot_fov_as_rectangle_mpl(
        fov_dict, ax, fc="none", ec="g", alpha=0.5
    )
    ax.autoscale()
    return {
        "plate polygon collection": plate_polycoll,
        "crystal polygon collection": crystal_polycoll,
        "fov polygon collection": fov_polycoll,
    }
