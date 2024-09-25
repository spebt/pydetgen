from .geometry import *
import numpy as np


def default_config():
    det_geoms = get_detector_units(0.475, np.array([[1.2, 8, 0]]), np.array([3, 2, 1]))
    plate_geoms = get_plate(3.5, 0, 16, np.array([2]), np.array([8]), 0, 1, 1)
    config_detector = {
        "detector": {
            "detector geometry": np.append(plate_geoms,det_geoms,axis=0).tolist(),
            "N subdivision xyz": [1, 1, 1],
            "active geometry indices": [1],
        }
    }

    config_fov = {
        "FOV": {
            "N voxels xyz": [50, 50, 1],
            "mm per voxel xyz": [1.0, 1.0, 1.0],
            "N subdivision xyz": [1, 1, 1],
        }
    }

    config_relation = {
        "relation": {
            "radial shift": {"format": "list", "data": [90]},
            "tangential shift": {"format": "list", "data": [0]},
            "rotation": {"format": "list", "data": [0]},
        }
    }
    return {**config_detector, **config_fov, **config_relation}
