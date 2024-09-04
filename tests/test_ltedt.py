import numpy as np
import pytest
import scipy.ndimage

import ltedt


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
        vol = create_test_volume((150, 100, 50), boundary=0.1)
        img = create_test_volume((50, 50, 1), frame=False).squeeze()
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

    return vol.squeeze()


def local_thickness_ref(data: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    distance_field_f64 = ndimage.distance_transform_edt(data)
    assert isinstance(distance_field_f64, np.ndarray)

    max_radius = distance_field_f64.max().astype(int)
    if max_radius <= np.iinfo(np.uint8).max:
        initial_distance_field = distance_field_f64.astype(np.uint8)
    elif max_radius <= np.iinfo(np.uint16).max:
        initial_distance_field = distance_field_f64.astype(np.uint16)
    elif max_radius <= np.iinfo(np.uint32).max:
        initial_distance_field = distance_field_f64.astype(np.uint32)
    else:
        initial_distance_field = distance_field_f64.astype(np.uint64)

    del distance_field_f64

    dilated = initial_distance_field.copy()
    for r in range(2, max_radius + 1):
        df = ndimage.distance_transform_edt(initial_distance_field < r)
        assert isinstance(df, np.ndarray)

        dilated[df < r] = r
    return dilated


@pytest.mark.parametrize(
    "data",
    [
        create_test_volume((200, 200), seed=42),
        create_test_volume((50, 50, 50), seed=42),
        create_test_volume((100, 100, 100), seed=42),
    ],
)
@pytest.mark.parametrize(
    "implementation",
    ["edt", "scipy", "cupy"],
)
def test_local_thickness_edt(data, implementation):
    thickness_ref = local_thickness_ref(data)
    thickness = ltedt.local_thickness(data, implementation=implementation, parallel=2)
    np.testing.assert_array_equal(thickness_ref, thickness)
