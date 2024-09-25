import numpy as np


def checkerboard(


    startxy: tuple[float, float], pitchxy: tuple[float, float], nxy: tuple[int, int]
) -> np.ndarray:
    """
    Generate checkerboard pattern centroid coordinates.

    Parameters
    ----------
    startxy : tuple of float
        Starting point of the checkerboard pattern (x, y).
    pitchxy : tuple of float
        Pitch of the checkerboard pattern (x, y).
    nxy : tuple of int
        Number of checkerboard squares in the x and y directions.

    Returns
    -------
    np.ndarray
        Array of checkerboard centers with shape (3, n), where n is the number of centers.


    """

    firstrow = np.array(
        np.meshgrid(startxy[0], np.arange(nxy[1]//2) * pitchxy[1] * 2 + startxy[1], 0)
    ).reshape(3, -1)
    pos = []
    for irow in range(nxy[0]):
        pos.append(firstrow.T + np.array([irow * pitchxy[0], (irow % 2) * pitchxy[0], 0]))
    return np.array(pos).T.reshape(3, -1)
