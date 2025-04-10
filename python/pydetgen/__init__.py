"""
PyDetGen: Python Detector Geometry Generator
"""

__all__ = [
    "get_polygon_vertices_sorted_by_xy_batch",
    "get_polygon_vertices_sorted_by_y_batch",
    "get_vertices_sorted_by_rad_raw_2d_batch",
    "get_polygon_vertices_sorted_by_rad_batch",
    "plot_polygons_from_vertices_mpl",
    "plot_fov_as_rectangle_mpl",
    "get_scanner_geometry_from_svg",
    "rotate_polygons",
    "set_plate_aperture_width",
    "get_df_rows_from_polygon_tensor",
    "get_polygon_groups_from_csv",
    "get_valid_cvs_path",
    "get_plate_aperture_center",
]

from .geometry_2d_helper import (
    get_polygon_vertices_sorted_by_xy_batch,
    get_polygon_vertices_sorted_by_y_batch,
    get_vertices_sorted_by_rad_raw_2d_batch,
    get_polygon_vertices_sorted_by_rad_batch,
    plot_polygons_from_vertices_mpl,
    plot_fov_as_rectangle_mpl,
    rotate_polygons,
    set_plate_aperture_width,
    get_df_rows_from_polygon_tensor,
    get_polygon_groups_from_csv,
    get_plate_aperture_center,
)
from .scanner_svg_helper import (
    get_scanner_geometry_from_svg,
)

from ._utils import get_valid_cvs_path
