import numpy as np
import itertools
from joblib import Parallel, delayed
import tqdm
from numba import jit
from typing import List, Callable, Tuple, Union, Optional
from numpy.typing import NDArray


@jit(nopython=True, cache=True)
def get_searchlight_data(data: NDArray[np.floating], coords: NDArray[np.int64]) -> NDArray[np.floating]:
    """Return all voxel data for a list of sphere coordinates."""
    # Get coordinates of all voxels in sphere
    out = np.zeros((len(coords), len(coords[0]), data.shape[1]))
    for i, sphere_coords in enumerate(coords):
        for j, coord in enumerate(sphere_coords):
            out[i, j] = data[coord]

    return out


class SearchLight:
    """
    SearchLight analysis with cross-validation and specified classifier.
    """

    def __init__(
        self,
        data: NDArray[np.floating],
        sl_fn: Callable[[NDArray[np.floating]], Union[float, NDArray[np.floating]]],
        radius: int,
        mask: Optional[NDArray[np.int_]] = None,
    ) -> None:
        """
        Parameters
        ----------
        radius : int
            Radius of searchlight in voxels.
        data : numpy.ndarray
            4D array with data.
        sl_fn : function
            Function that takes data and returns a single value for each voxel.
        mask : numpy.ndarray, optional
            3D array with mask. If specified, only voxels with mask==1 will be used.
        """

        self.radius = radius
        self.original_data_shape = data.shape
        self.data = data.reshape(-1, data.shape[-1])
        self.sl_fn = sl_fn
        self.mask = mask

        # pre-compile get_searchlight_data (not sure if this does anything, haha)
        _ = get_searchlight_data(self.data, np.array([self.get_sphere_coords(0, 0, 0)]))

    def fit(
        self,
        coords: Optional[List[Tuple[int, int, int]]] = None,
        output_size: int = 1,
        n_jobs: int = 1,
        n_chunks: int = 1,
        verbose: int = 1,
    ) -> NDArray[np.floating]:
        """
        Returns a 3D array with the classification accuracy for each voxel.

        Parameters
        ----------
        coords : list of tuples, optional
            List of coordinates to use as searchlight centers. If not specified, all voxels will be used.
        output_size : int, optional
            Number of output values per voxel - allows for multiple outputs per voxel (e.g. for running multiple classifiers for each voxel).
        n_jobs : int, optional
            Number of jobs to run in parallel.
        n_chunks : int, optional
            Number of chunks to split data into for data preloading, to avoid memory issues.
        verbose : int, optional
            Verbosity level.
        """
        x_shape, y_shape, z_shape = self.original_data_shape[:3]

        if coords is not None:
            coords = np.array(coords)
        elif self.mask is not None:
            print("Using mask to determine searchlight coordinates")
            coords = list(zip(*np.nonzero(self.mask)))
        else:
            print("Neither mask nor coordinates specified, using all voxels")
            coords = itertools.product(range(x_shape), range(y_shape), range(z_shape))

        results = np.zeros((x_shape, y_shape, z_shape, output_size))

        # split data into chunks for data preloading
        coord_results = []

        print("Loading sphere indices")
        sphere_coords = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self.get_sphere_coords)(*coords_) for coords_ in coords
        )
        chunks = np.array_split(list(sphere_coords), n_chunks)

        for chunk_i, chunk in enumerate(chunks):
            print(f"Chunk {chunk_i+1}/{len(chunks)}")
            print("Loading chunk data")
            data = np.moveaxis(get_searchlight_data(self.data, chunk), 1, -1)
            print("Running searchlight")
            if n_jobs == 1:
                for data_ in tqdm.tqdm(data):
                    coord_results.append(self.sl_fn(data_))
            else:
                coord_results.extend(
                    Parallel(n_jobs=n_jobs, verbose=5)(
                        delayed(self.sl_fn)(data_) for data_ in data
                    )
                )

        for i, index in enumerate(coords):
            results[index] = coord_results[i]

        return results

    def get_sphere_coords(self, x: int, y: int, z: int) -> NDArray[np.int_]:
        """
        Returns coordinates of all voxels in a sphere with radius self.radius
        at position x, y, z
        """
        # Get coordinates of all voxels in cube with side length 2*self.radius+1
        x_coords, y_coords, z_coords = np.mgrid[
            x - self.radius : x + self.radius + 1,
            y - self.radius : y + self.radius + 1,
            z - self.radius : z + self.radius + 1,
        ]
        # Get coordinates of all voxels in sphere
        sphere_coords = np.sqrt(
            (x_coords - x) ** 2 + (y_coords - y) ** 2 + (z_coords - z) ** 2
        )
        x_coords = x_coords[sphere_coords <= self.radius]
        y_coords = y_coords[sphere_coords <= self.radius]
        z_coords = z_coords[sphere_coords <= self.radius]

        coords = list(zip(x_coords, y_coords, z_coords))

        coords = np.ravel_multi_index(
            np.array(coords).T, self.original_data_shape[:3], mode="clip"
        )

        return coords
