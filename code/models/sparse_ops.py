# -*- coding: utf-8 -*-

"""
* Name:         src/utils/sparse_ops.py
* Description:  Calculate sparsity and perform inner product operations.
* Author:       Imed KERAGHEL.
* Created:      18/03/2024
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================


from typing import Union

from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import csr_matrix
from numpy import ndarray
import numpy as np


def check_sparsity(x: ndarray) -> float:
    """
    Calculate sparsity of centroid vectors using predefined functions.

    Args:
        x : Data to check.

    Returns:
        1 - proportion of nonzero elements
    """
    total_elements = x.size
    nonzero_elements = np.count_nonzero(x)

    return 1 - (nonzero_elements / total_elements)


def inner_product(X: Union[ndarray, csr_matrix], Y: Union[ndarray, csr_matrix]) -> csr_matrix:
    """
    Compute inner product between two arrays.

    Args:
    X, Y: numpy.ndarray or scipy.sparse.csr_matrix.
        One of both must be sparse matrix
        shape of X = (n,p)
        shape of Y = (p,m)

    Returns
        scipy.sparse.csr_matrix shape of Z = (n, m)
    """
    return safe_sparse_dot(X, Y, dense_output=False)
