import datetime
import numpy as np
from pydetgen.objects import ConfigBlock

_INDENT_N_SPACES = 2
_INDENT_STR = " " * _INDENT_N_SPACES


def dump_array_1D(data: np.ndarray, level: int = 0) -> str:
    indent = _INDENT_STR * level
    return indent + np.array2string(data, separator=", ")


def dump_array_2D(data: np.ndarray, level: int = 0) -> str:
    rows = []
    indent = _INDENT_STR * level
    for row in data:
        rows.append(indent + _INDENT_STR + np.array2string(row, separator=", "))

    output = indent + "[\n"
    output += ",\n".join(rows)
    output += "\n" + indent + "]"
    return output


def dump_block(obj: ConfigBlock, level: int = 0) -> str:
    indent = _INDENT_STR * level
    output = indent + obj.objname + ":"

    if obj.datatype == "array 1D":
        assert len(obj.data.shape) == 1
        output += "\n"
        output += dump_array_1D(obj.data, level + 1)
    elif obj.datatype == "array 2D":
        assert len(obj.data.shape) == 2
        output += "\n"
        output += dump_array_2D(obj.data, level + 1)
    elif obj.datatype == "object":
        output += "\n"
        blocks = [dump_block(block_obj, level + 1) for block_obj in obj.data]
        output += "\n".join(blocks)
    elif obj.datatype in ["string", "number", "integer", "boolean"]:
        output += " " + str(obj.data)
    else:
        raise ValueError(f"Unknown data type: {obj.datatype}")
    return output


def write(
    fname: str,
    plate_geoms,
    det_geoms,
    active_det_idx,
    det_nsubs,
    img_nsubs,
    nvx,
    mmpvx,
    dist,
    rotation_deg,
):
    active_det_idx_str = "[" + ", ".join(str(el) for el in active_det_idx) + "]"
    indent_level_N_spaces = range(0, 10, 2)
    geom_lines_level = 3
    now_str = datetime.datetime.now().strftime("%x %X")
    plate_geoms_lines = []
    for geoms in plate_geoms:
        plate_geoms_lines.append("[" + ", ".join(str(el) for el in geoms) + "]")
    plate_geoms_line_block = ""
    if len(plate_geoms_lines) != 0:
        plate_geoms_line_block = (
            " " * indent_level_N_spaces[geom_lines_level]
            + (",\n" + " " * indent_level_N_spaces[geom_lines_level]).join(
                el for el in plate_geoms_lines
            )
            + ",\n"
        )
    det_geoms_lines = []
    for geoms in det_geoms:
        det_geoms_lines.append("[" + ", ".join(str(el) for el in geoms) + "]")

    det_geoms_line_block = " " * indent_level_N_spaces[geom_lines_level] + (
        ",\n" + " " * indent_level_N_spaces[geom_lines_level]
    ).join(el for el in det_geoms_lines)

    with open(fname, "w") as fout:
        fout.write(
            "# This is an automatically generated systematic matrix configuration file in YAML format\n"
        )
        fout.write("# Generated on: " + now_str + "\n")
        fout.write("detector:\n")
        fout.write(" " * indent_level_N_spaces[1] + "detector geometry:\n")
        fout.write(
            " " * indent_level_N_spaces[2] + "# detector geometry in millimeter.\n"
        )
        fout.write(
            " " * indent_level_N_spaces[2]
            + "# Defined in cuboids with x_0, x_1, y_0, y_1, z_0, z_1\n"
        )
        fout.write(
            " " * indent_level_N_spaces[2]
            + "# parameter 0 to 1: radial coordinates, x_0, x_1\n"
        )
        fout.write(
            " " * indent_level_N_spaces[2]
            + "# parameter 2 to 3: tangential coordiantes, y_0, y_1\n"
        )
        fout.write(
            " " * indent_level_N_spaces[2]
            + "# parameter 4 to 5: axial coordiantes, z_0, z_1\n"
        )
        fout.write(
            " " * indent_level_N_spaces[2]
            + "# # parameter 6: cuboid type identifier.\n"
        )
        fout.write(
            " " * indent_level_N_spaces[2]
            + "# 0 is non-detector, 1 and greater numbers are sequential indices for detector units.\n"
        )
        fout.write(
            " " * indent_level_N_spaces[2]
            + "# parameter 7: cuboid attenuation coefficient\n"
        )
        fout.write(" " * indent_level_N_spaces[2] + "[\n")
        if len(plate_geoms_lines) != 0:
            fout.write(plate_geoms_line_block)
        fout.write(det_geoms_line_block)
        fout.write("\n" + " " * indent_level_N_spaces[2] + "]\n")
        fout.write(
            " " * indent_level_N_spaces[1]
            + "N-subdivision xyz: [%d, %d, %d]"
            % (det_nsubs[0], det_nsubs[1], det_nsubs[2])
            + "\n"
        )
        fout.write(
            " " * indent_level_N_spaces[1]
            + "active geometry indices: %s\n" % active_det_idx_str
        )
        fout.write(" " * indent_level_N_spaces[0] + "# Image space parameters\n")
        fout.write(" " * indent_level_N_spaces[0] + "image:\n")
        fout.write(
            " " * indent_level_N_spaces[1]
            + "N-voxels xyz: [%d, %d, %d]\n" % (nvx[0], nvx[1], nvx[2])
        )
        fout.write(
            " " * indent_level_N_spaces[1]
            + "mm-per-voxel xyz: [%d, %d, %d]\n" % (mmpvx[0], mmpvx[1], mmpvx[2])
        )
        fout.write(
            " " * indent_level_N_spaces[1]
            + "N-subdivision xyz: [%d, %d, %d]\n"
            % (img_nsubs[0], img_nsubs[1], img_nsubs[2])
        )
        fout.write(
            " " * indent_level_N_spaces[0]
            + "# Image space to detector space relative positioning\n"
        )
        fout.write(" " * indent_level_N_spaces[0] + "detector-to-image:\n")
        fout.write(
            " " * indent_level_N_spaces[1]
            + "# detector front edge to FOV center distance in radial direction\n"
        )
        fout.write(
            " " * indent_level_N_spaces[1] + "# acceptable units are mm, cm, m\n"
        )
        fout.write(
            " " * indent_level_N_spaces[1] + "radial distance: %s mm\n" % str(dist)
        )
        fout.write(
            " " * indent_level_N_spaces[1]
            + "# rotation of the detector relative to the FOV in degrees\n"
        )
        fout.write(
            " " * indent_level_N_spaces[1] + "rotation: %s\n" % str(rotation_deg)
        )
