from functools import lru_cache

import numpy as np
import pytest

import ltedt
from ltedt.util import create_test_volume


@lru_cache
def local_thickness_ref(shape, sigma, seed):
    import edt

    data = create_test_volume(shape, sigma=sigma, seed=seed)

    distance_field_f32 = edt.edt(data, parallel=2)

    max_radius = distance_field_f32.max().astype(int)
    if max_radius <= np.iinfo(np.uint8).max:
        initial_distance_field = distance_field_f32.astype(np.uint8)
    elif max_radius <= np.iinfo(np.uint16).max:
        initial_distance_field = distance_field_f32.astype(np.uint16)
    elif max_radius <= np.iinfo(np.uint32).max:
        initial_distance_field = distance_field_f32.astype(np.uint32)
    else:
        initial_distance_field = distance_field_f32.astype(np.uint64)

    del distance_field_f32

    dilated = initial_distance_field.copy()
    for r in range(2, max_radius + 1):
        df = edt.edtsq(initial_distance_field < r, parallel=2)
        dilated[df < r * r] = r

    return data, dilated


TEST_DATA_PARAMS = [
    ((200, 200), 10, 42),
    ((1000, 1000), 20, 42),
    ((50, 50, 50), 7, 42),
    ((250, 250, 250), 14, 42),
    # ((500, 500, 500), 20, 42),
]

TEST_FUNCTION_PARAMS = [
    ("edt", 1),
    ("edt", 2),
    ("edt", 4),
    ("scipy", 1),
    ("cupy", 1),
]


@pytest.mark.parametrize("data_params", TEST_DATA_PARAMS)
@pytest.mark.parametrize("implementation,parallel", TEST_FUNCTION_PARAMS)
def test_local_thickness(data_params, implementation, parallel, benchmark):
    shape, sigma, seed = data_params
    data, thickness_ref = local_thickness_ref(shape, sigma=sigma, seed=seed)
    thickness = benchmark(ltedt.local_thickness, data, implementation, parallel)
    np.testing.assert_array_equal(thickness_ref, thickness)
