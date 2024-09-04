from typing import Literal

import numpy as np


def local_thickness_edt(data: np.ndarray, parallel: int = 1):
    import edt

    distance_field_f32 = edt.edt(data, parallel=parallel)

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
        df = edt.edtsq(initial_distance_field <= r, parallel=parallel)
        dilated[df <= r * r] = r
    return dilated


def local_thickness_scipy(data: np.ndarray) -> np.ndarray:
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
        df = ndimage.distance_transform_edt(initial_distance_field <= r)
        assert isinstance(df, np.ndarray)

        dilated[df <= r] = r
    return dilated


def local_thickness_cupy(data: np.ndarray) -> np.ndarray:
    import cupy as cp
    from cupyx.scipy import ndimage as ndimage_cp

    data_cp = cp.asarray(data)
    distance_field_f32_cp = cp.empty(data_cp.shape, dtype=cp.float32)
    ndimage_cp.distance_transform_edt(data_cp, float64_distances=False, distances=distance_field_f32_cp)
    max_radius = distance_field_f32_cp.max().astype(int)

    if max_radius <= cp.iinfo(cp.uint8).max:
        initial_distance_field_cp = distance_field_f32_cp.astype(cp.uint8)
    elif max_radius <= cp.iinfo(cp.uint16).max:
        initial_distance_field_cp = distance_field_f32_cp.astype(cp.uint16)
    elif max_radius <= cp.iinfo(cp.uint32).max:
        initial_distance_field_cp = distance_field_f32_cp.astype(cp.uint32)
    else:
        initial_distance_field_cp = distance_field_f32_cp.astype(cp.uint64)

    dilated_cp = initial_distance_field_cp.copy()
    for r in cp.arange(2, max_radius + 1, dtype=dilated_cp.dtype):
        ndimage_cp.distance_transform_edt(
            initial_distance_field_cp <= r,
            float64_distances=False,
            distances=distance_field_f32_cp,
        )
        dilated_cp[distance_field_f32_cp <= r] = r
    dilated = cp.asnumpy(dilated_cp)

    return dilated


def local_thickness(
    data: np.ndarray,
    implementation: Literal["edt", "scipy", "cupy"] = "edt",
    parallel: int = 1,
) -> np.ndarray:
    if implementation == "edt":
        return local_thickness_edt(data, parallel=parallel)
    elif implementation == "scipy":
        return local_thickness_scipy(data)
    elif implementation == "cupy":
        return local_thickness_cupy(data)
    else:
        raise ValueError(f"Invalid implementation: {implementation}")
