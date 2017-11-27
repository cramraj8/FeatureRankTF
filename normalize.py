# -*- coding: utf-8 -*-

from __future__ import division, absolute_import, print_function
import numpy as np


def z_score_normalization(data, axis=0):
    r"""Function for normalizing the feature vectors.
    Normalizes each feature vectors from the input matrix with
    zero mean and unit variance.
    normalized_features = (data - feature_mean) / feature_std
    Parameters
    ----------
    data : numpy.ndarray
        A M * N array containing feature vectors.
    axis : int
        A binary variable indicating along which axis to perform the operation.
    Returns
    -------
    data : numpy.ndarray
        A standard-normalized dataset.
    """
    data = np.asarray(data, np.float32)

    if (axis == 1):
        data = data.transpose()

    # Normalize the matrix using positive standard deviation.
    data = (data - np.nanmean(data,
                              axis=0,
                              dtype=np.float32)) / np.nanstd(data,
                                                             axis=0,
                                                             dtype=np.float32)

    return data
