# Fast local thickness using Euclidean distance transform
A Python package for calculating local thickness in 2D or 3D using Euclidean distance transform.

## Installation
Not available in PyPI yet. You can install directly form repository.
```
pip install git+https://github.com/Skielex/ltedt.git
```

## How to use
``` python
from ltedt import local_thickness

# `data` is a 2D or 3D numpy bool array (i.e., binary segmentation).
thickness = local_thickness(data)
# `thickness` is a integer array with local thicknesses of the same shape of the input.

# To run the calculations on four CPU cores.
thickness = local_thickness(data, parallel=4)

# To run the calculations on a CUDA device.
thickness = local_thickness(data, implementation="cupy")
```

## Different implementations
The `local_thickness` function support three different implementations of EDT based on the following packages:
- [edt](https://github.com/seung-lab/euclidean-distance-transform-3d/) - This is the default implementation used by `local_thickness`. Specifically, the `edt` and `eqtsq` functions of the package are used. These allow for using multiple threads by settings the `parallel` argument, which allows them to considerably faster than `scipy.ndimage.distance_transform_edt`.
- [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html) - This is an optional alternative to the `edt` package, which uses the `scipy.ndimage.distance_transform_edt` function.
- [cupy](https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.ndimage.distance_transform_edt.html) - This is an optional alternative, which uses `cupyx.scipy.ndimage.distance_transform_edt` to calculate the local thickness using a CUDA device (e.g., Nvidia GPU). This is usually **very** fast and can speed up computation by over 100x on large volumes compared to the CPU-based implementations. However, the available GPU VRAM will limit how large images/volumes can be operated on.

Which EDT implementation to use is determined by the `implementation` parameter of the `local_thickness` function.