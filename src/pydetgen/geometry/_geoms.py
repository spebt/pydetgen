import numpy as np


def get_plate(
    mu: float,
    start_y: float,
    width_y: float,
    aperture_width: np.ndarray | float,
    aperture_pos: np.ndarray,
    start_x: float,
    thickness_x: float,
    height_z: float,
) -> np.ndarray:
    """
    Generate the geometry of a plate with apertures.

    Parameters
    ----------
    mu : float
        A parameter related to the material property of the plate.
    start_y : float
        The starting y-coordinate of the plate.
    width_y : float
        The width of the plate in the y-direction.
    aperture_width : np.ndarray or float
        A 1-D array of aperture widths. Each element represents the width of an aperture in y-direction.
    aperture_pos : np.ndarray
        A 1-D array of y-coordinates for the centers of the apertures.
    start_x : float
        The starting x-coordinate of the plate.
    thickness_x : float
        The thickness of the plate in the x-direction.
    height_z : float
        The height of the plate in the z-direction.

    Returns
    -------
    np.ndarray
        A 2-D array representing the geometry of the plate with apertures. Each row corresponds to a segment of the plate.

    Raises
    ------
    AssertionError
        If any apertures overlap, an assertion error is raised.
    """

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
    return np.round(geoms, decimals=6)


def get_detector_units(
    mu: np.ndarray | float, pos: np.ndarray, dims: np.ndarray
) -> np.ndarray:
    """
    Calculate the detector units based on the provided positions, dimensions, and mu values.

    Parameters
    ----------
    mu : np.ndarray or float
        An array of attenuation coefficients or a single attenuation coefficient for all detector units.
    pos : np.ndarray
        An array of detector unit positions with shape (N, 3).
    dims : np.ndarray
        An array of detector unit dimensions which can be of shape (3,) or (N, 3).
        - If the shape is (3,), the same dimensions are used for all detector units.
        - If the shape is (N, 3), the dimensions are used for each detector unit.

    Returns
    -------
    np.ndarray
        A numpy array of shape (N, 8) containing the calculated detector units.

    Raises
    ------
    Exception
        If the shape of `dims` is not (3,) or (N, 3).
    """
    if pos.shape[0] != dims.shape[0] and dims.shape[0] != 3:
        raise Exception("Dimension input is in wrong shape.")
    else:
        indice = np.arange(pos.shape[0]) + 1
        geoms = np.empty((pos.shape[0], 8), dtype=np.float64)
        geoms[:, 6] = indice
        if dims.shape[0] == 3:
            geoms[:, :6:2] = pos[:] - dims * 0.5
            geoms[:, 1:6:2] = pos[:] + dims * 0.5
        elif pos.shape == dims.shape:
            geoms[:, :6:2] = pos[:] - dims[:] * 0.5
            geoms[:, 1:6:2] = pos[:] + dims[:] * 0.5
        geoms[:, 7] = mu
        return np.round(geoms, decimals=6)
