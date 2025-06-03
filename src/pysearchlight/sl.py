import numpy as np
import itertools
from joblib import Parallel, delayed
import tqdm
from numba import jit
from typing import List, Callable, Tuple


@jit(nopython=True, cache=True)
def get_searchlight_data(data, coords):
    """Return the data for each searchlight sphere.

    Parameters
    ----------
    data : np.ndarray
        Flattened data array of shape ``(n_voxels, n_features)``.
    coords : np.ndarray
        Iterable of arrays containing voxel indices for every searchlight
        sphere.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_centers, n_sphere_voxels, n_features)`` containing
        the extracted data for every searchlight center.
    """
    out = np.zeros((len(coords), len(coords[0]), data.shape[1]))
    for i, sphere_coords in enumerate(coords):
        for j, coord in enumerate(sphere_coords):
            out[i, j] = data[coord]

    return out


class SearchLight:
    """Perform a searchlight analysis on 4D data.

    The class takes care of iterating a spherical region across the provided
    data and applying a user supplied function to each sphere.

    Parameters
    ----------
    data : np.ndarray
        Input array with shape ``(x, y, z, samples)``.
    sl_fn : Callable
        Function that receives the data within a sphere as
        ``(n_voxels_in_sphere, n_samples)`` and returns one or more values.
    radius : int
        Radius of the spherical searchlight in voxels.
    mask : np.ndarray, optional
        Optional boolean mask of shape ``(x, y, z)`` limiting the voxels that
        are evaluated.
    """

    def __init__(
        self, data: np.array, sl_fn: Callable, radius: int, mask: np.array = None
    ):
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
        coords: List[Tuple] = None,
        output_size: int = 1,
        n_jobs: int = 1,
        n_chunks: int = 1,
        verbose: int = 1,
    ) -> np.array:
        """Run the searchlight analysis.

        Parameters
        ----------
        coords : list of tuple of int, optional
            Specific coordinates to use as centers of the searchlight. When not
            provided all voxels (or the provided mask) are used.
        output_size : int, optional
            Expected number of values returned by ``sl_fn`` for each voxel.
        n_jobs : int, optional
            Number of parallel jobs.
        n_chunks : int, optional
            Split the computation into this many chunks in order to reduce
            memory consumption.
        verbose : int, optional
            Verbosity level forwarded to ``joblib``.

        Returns
        -------
        np.ndarray
            Array of shape ``(x, y, z, output_size)`` with the computed result
            for each voxel.
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

    def get_sphere_coords(self, x, y, z):
        """Return voxel indices for the sphere centered at ``(x, y, z)``."""
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
