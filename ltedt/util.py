import numpy as np
import scipy.ndimage


def create_test_volume(
    shape: tuple[int, int, int] | tuple[int, int],
    sigma: float = 7,
    threshold: float = 0,
    boundary: float = 0,
    frame: bool = True,
    seed: int | None = None,
):
    """Creates test volume for local thickness and porosity analysis.
    Arguments:
        shape: tuple giving the size of the volume
        sigma: smoothing scale, higher value - smoother objects
        threshold: a value close to 0, larger value - less material (smaller objects)
        boundary: strength of imposing object boundary pulled inwards
        frame: one-voxel frame of False
    Returns:
        a test volume
    Example uses:
        vol = create_test_volume((150, 100, 50))
        img = create_test_volume((50, 50))
    Authors: vand@dtu.dk, 2019, niejep@dtu.dk, 2024
    """

    if len(shape) == 3:
        r = np.fromfunction(
            lambda x, y, z: (
                (x / (shape[0] - 1) - 0.5) ** 2 + (y / (shape[1] - 1) - 0.5) ** 2 + (z / (shape[2] - 1) - 0.5) ** 2
            )
            ** 0.5,
            shape,
            dtype=int,
        )
    elif len(shape) == 2:
        r = np.fromfunction(
            lambda x, y: ((x / (shape[0] - 1) - 0.5) ** 2 + (y / (shape[1] - 1) - 0.5) ** 2) ** 0.5,
            shape,
            dtype=int,
        )

    prng = np.random.RandomState(seed)  # pseudo random number generator
    vol = prng.standard_normal(shape)
    vol[r > 0.5] -= boundary
    vol = scipy.ndimage.gaussian_filter(vol, sigma, mode="constant", cval=-boundary)
    vol = vol > threshold
    if frame:
        vol[[0, -1]] = False
        vol[:, [0, -1]] = False
        if len(shape) == 3:
            vol[:, :, [0, -1]] = False

    return vol
