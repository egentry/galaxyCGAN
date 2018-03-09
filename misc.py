import numpy as np


def transform_0_1(X):
    return (X - X.min())/(X.max() - X.min())


def y_binner(y):
    """Converts continuous y ({z, M_stellar}) into binned values

    Inputs
    ------
    y - 1d array-like, size 2
      - should correspond to [redshift (z), log_10 stellar mass / M_sun]

    Outputs
    -------
    new_y - np.ndarray (ndim = 1, size = 9, dtype=float)
          - two one-hot vectors, corresponding to the binned y_value,
            for redshift (entries 0-4) and stellar mass (entries 5-8)
    """
    z = y[0]
    M_stellar = y[1]

    z_bin_edges = np.array([0, .05, 0.10, 0.15, 0.20, 10])
    M_bin_edges = np.array([0, 8, 8.5, 9, np.inf])

    # should I make this size Mbins + z_bins or M_bins * z_bins?
    # let's start with the smaller one.
    new_y = np.zeros((M_bin_edges.size - 1) + (z_bin_edges.size - 1))

    new_y[:z_bin_edges.size-1], _ = np.histogram([z],
                                                 bins=z_bin_edges)

    new_y[z_bin_edges.size-1:], _ = np.histogram([M_stellar],
                                                 bins=M_bin_edges)

    return new_y
