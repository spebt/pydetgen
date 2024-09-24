import numpy as np


def get_plate_geoms_2d(
    mu: float,
    start_y: float,
    width_y: float,
    aperture_width: np.ndarray,
    aperture_pos: np.ndarray,
    start_x: float,
    thickness_x: float,
    height_z: float,
) -> np.ndarray:
    
    # apperture_pos is a 1-D Numpy.ndarray consisting of the y-coordinates of the aperture centers.
    # apperture_width is a 1-D Numpy.ndarray consisting of the aperture width. They can be different.
    n_aperture = aperture_pos.shape[0]
    ycoords = np.empty(n_aperture * 2, dtype=np.float64)
    ycoords[::2] = -aperture_width * 0.5 + aperture_pos
    ycoords[1::2] = aperture_width * 0.5 + aperture_pos
    ycoords = np.append(np.insert(ycoords, 0, start_y), start_y + width_y)
    assert (
        np.unique(ycoords).shape[0] == ycoords.shape[0]
    ), "Apertures overlapped, check the inputs!"
    geoms = np.empty((n_aperture + 1, 8), dtype=np.float64)
    geoms[:, 0] = start_x
    geoms[:, 1] = start_x + thickness_x
    geoms[:, 4:6] = np.array([-height_z * 0.5, height_z * 0.5])
    geoms[:, 2] = ycoords[::2]
    geoms[:, 3] = ycoords[1::2]
    geoms[:, 6] = 0
    geoms[:, 7] = mu
    return np.round(geoms,decimals=6)


def get_det_geoms(mu: np.ndarray, pos: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """
    dims: can be of shape (3,) or (N points,3)
    """
    if pos.shape[0] != dims.shape[0] and dims.shape[0] != 3:
        raise Exception("Dimension input is in wrong shape.")
    else:
        indice = np.arange(pos.shape[0]) + 1
        geoms = np.empty((pos.shape[0], 8),dtype=np.float64)
        geoms[:, 6] = indice
        if dims.shape[0] == 3:
            geoms[:, :6:2] = pos[:] - dims * 0.5
            geoms[:, 1:6:2] = pos[:] + dims * 0.5
        elif pos.shape == dims.shape:
            geoms[:, :6:2] = pos[:] - dims[:] * 0.5
            geoms[:, 1:6:2] = pos[:] + dims[:] * 0.5
        geoms[:, 7] = mu
        return np.round(geoms,decimals=6)
