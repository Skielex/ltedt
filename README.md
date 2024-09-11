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

## Benchmark results
The tables below contains benchmark results for all three EDT implementations (including varying thread counts for `edt`). The results show that the implementation doesn't matter much for smaller datasets, but has varries by over 100x for larger datasets.

Benchmarks were run on an AMD Ryzen 9 5900X 12-Core CPU and Nvidia RTX 2700 SUPER GPU running Windows 11. Initial testing on a Nvidia A100 GPU indicates upwards of a 1000x speed-up compared to using the CPU implementations on large volumes (larger than 1000³).

### shape: (200, 200), sigma: 10
| Benchmark name       |   Med time (s) |   Relative |   Min time (s) |   Max time (s) |   Mean time (s) |
|:---------------------|---------------:|-----------:|---------------:|---------------:|----------------:|
| edt-2-data_params0   |          0.017 |          1 |          0.017 |          0.019 |           0.017 |
| edt-4-data_params0   |          0.018 |          1 |          0.018 |          0.019 |           0.018 |
| cupy-1-data_params0  |          0.018 |          1 |          0.017 |          0.021 |           0.019 |
| edt-1-data_params0   |          0.019 |          1 |          0.019 |          0.020 |           0.019 |
| scipy-1-data_params0 |          0.020 |          1 |          0.020 |          0.022 |           0.021 |
### shape: (1000, 1000), sigma: 20
| Benchmark name       |   Med time (s) |   Relative |   Min time (s) |   Max time (s) |   Mean time (s) |
|:---------------------|---------------:|-----------:|---------------:|---------------:|----------------:|
| cupy-1-data_params1  |          0.054 |          1 |          0.053 |          0.059 |           0.055 |
| edt-4-data_params1   |          0.598 |         11 |          0.596 |          0.600 |           0.598 |
| edt-2-data_params1   |          0.674 |         13 |          0.670 |          0.682 |           0.675 |
| edt-1-data_params1   |          0.834 |         16 |          0.827 |          0.843 |           0.833 |
| scipy-1-data_params1 |          1.704 |         32 |          1.694 |          1.706 |           1.702 |
### shape: (50, 50, 50), sigma: 7
| Benchmark name       |   Med time (s) |   Relative |   Min time (s) |   Max time (s) |   Mean time (s) |
|:---------------------|---------------:|-----------:|---------------:|---------------:|----------------:|
| cupy-1-data_params2  |          0.008 |          1 |          0.008 |          0.019 |           0.010 |
| edt-2-data_params2   |          0.052 |          6 |          0.051 |          0.055 |           0.053 |
| edt-4-data_params2   |          0.054 |          7 |          0.051 |          0.058 |           0.054 |
| edt-1-data_params2   |          0.065 |          8 |          0.064 |          0.067 |           0.065 |
| scipy-1-data_params2 |          0.079 |         10 |          0.077 |          0.081 |           0.079 |
### shape: (250, 250, 250), sigma: 14
| Benchmark name       |   Med time (s) |   Relative |   Min time (s) |   Max time (s) |   Mean time (s) |
|:---------------------|---------------:|-----------:|---------------:|---------------:|----------------:|
| cupy-1-data_params3  |          0.229 |          1 |          0.227 |          0.233 |           0.229 |
| edt-4-data_params3   |          7.137 |         31 |          7.097 |          7.270 |           7.159 |
| edt-2-data_params3   |          9.105 |         40 |          9.074 |          9.143 |           9.106 |
| edt-1-data_params3   |         14.129 |         62 |         14.036 |         14.234 |          14.122 |
| scipy-1-data_params3 |         36.873 |        161 |         36.793 |         36.895 |          36.861 |
### shape: (500, 500, 500), sigma: 20
| Benchmark name       |   Med time (s) |   Relative |   Min time (s) |   Max time (s) |   Mean time (s) |
|:---------------------|---------------:|-----------:|---------------:|---------------:|----------------:|
| cupy-1-data_params4  |          2.618 |          1 |          2.583 |          2.661 |           2.620 |
| edt-4-data_params4   |         78.074 |         30 |         77.908 |         78.275 |          78.065 |
| edt-2-data_params4   |        101.821 |         39 |        101.791 |        101.949 |         101.862 |
| edt-1-data_params4   |        157.758 |         60 |        157.393 |        158.106 |         157.719 |
| scipy-1-data_params4 |        528.846 |        202 |        524.870 |        534.517 |         528.780 |

## Related work
This work was inspired by the [`localthickness`](https://github.com/vedranaa/local-thickness) package.