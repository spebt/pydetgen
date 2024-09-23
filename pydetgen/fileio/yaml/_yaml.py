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


def dump(obj: ConfigBlock) -> str:
