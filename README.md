# PySearchLight - A simple and customizable implementation of searchlight analysis 🔍 

This package provides an easy-to-use implementation of the searchlight analysis by [Kriegeskorte et al. (2006)](https://www.pnas.org/doi/abs/10.1073/pnas.0600244103). In this analysis method, a spherical "searchlight" is moved across a 4D array (typically brain data) and a user-defined function is applied to the data within the searchlight. This allows for the computation of a measure of interest (e.g., classification accuracy) at each voxel location, based on the data within the sphere centered at that voxel.

## Installation

You can install PySearchLight with pip:

```bash
pip install pysearchlight
```

## Usage

PySearchlight takes care of moving the sphere across the data and applying the user-defined function to the data within the sphere. The user only needs to provide the data, the function to apply within the searchlight, and the radius of the searchlight sphere. The function should take a single argument, which is the data within the sphere centered at a voxel location. The function can also take additional arguments, which can be passed using `functools.partial`.

Here is a simple example of how to use PySearchLight to train and evaluate a classifier on data within a searchlight:

```python
import numpy as np
from pysearchlight import SearchLight

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from functools import partial

# Create some example data
data = np.random.rand(10, 10, 10, 100)

# Create a binary mask
mask = np.random.randint(0, 2, (10, 10, 10)) 

# Define a function to train a classifier on the data within a searchlight sphere
def train_classifier(data, labels):
    clf = SVC()
    scores = cross_val_score(clf, data, labels)
    return scores.mean()

# Because the searchlight function should only take one argument (the data in the sphere centered 
# at a voxel location), we use partial to pass the labels:
train_classifier_partial = partial(train_classifier, labels=np.random.randint(0, 2, 100))

# Create a SearchLight object
sl = SearchLight(
    data=data, # The 4D data array 
    sl_fn=train_classifier_partial, # The function to apply within the searchlight
    radius=2, # The radius of the searchlight sphere (in voxels)
    mask=mask, # A binary mask indicating which voxels to include in the searchlight
)

# Run the searchlight analysis (this will return a 3D array of results)
results = sl.fit(
    n_jobs=1, # The number of jobs to run in parallel
    n_chunks=10, # The number of chunks to split the data into (to reduce memory usage)
    verbose=1, # Verbosity level
)
```