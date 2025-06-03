import numpy as np
from pysearchlight.sl import SearchLight

def dummy(data):
    return 0

def test_get_sphere_coords_size_consistency():
    data = np.zeros((5, 5, 5, 1))
    sl = SearchLight(data=data, sl_fn=dummy, radius=1)
    center = sl.get_sphere_coords(2, 2, 2)
    edge = sl.get_sphere_coords(0, 0, 0)
    assert len(center) == len(edge)
