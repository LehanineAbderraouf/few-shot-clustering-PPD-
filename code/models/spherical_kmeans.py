# -*- coding: utf-8 -*-

"""
* Name:         src/spherical_kmeans.py
* Description:  Spherical k-Means clustering.
* Author:       Imed KERAGHEL.
* Created:      18/03/2024
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================


import datetime
import os
import numpy as np
import scipy.sparse as sp
import time
from typing import Union

from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils import as_float_array
from numpy import ndarray
from scipy.sparse import csr_matrix

from models.sparse_ops import check_sparsity
from models.sparse_ops import inner_product


class SphericalKMeans:
    """
    Spherical k-Means clustering.

    Parameters:
        n_clusters (int): The number of clusters to form as well as the number of centroids to generate. Defaults to 8.
        init (Union[str, ndarray]): Method for initialization, defaults to 'k-means++'. Can be one of
        ['similar_cut', 'k-means++', 'random'] or an ndarray of shape (n_clusters, n_features).
        sparsity (Optional[str]): Method for preserving sparsity of centroids. One of ['sculley', 'minimum_df', None].
        Defaults to None.
        max_iter (int): Maximum number of iterations of the k-means algorithm for a single run. Defaults to 10.
        tol (float): Relative tolerance in regard to inertia to declare convergence. Defaults to 1e-4.
        verbose (int): Verbosity mode. Defaults to 0.
        random_state (Optional[Union[int, RandomState]]): The seed used by the random number generator; a RandomState
        instance; or None. Defaults to None.
        debug_directory (Optional[str]): If not None, model saves logs, temporal cluster labels, and temporal cluster
        centroids for all iterations. Defaults to None.
        algorithm (Optional[str]): Computation algorithm. Currently, ignored. Defaults to None.
        max_similar (float): 'similar_cut initializer' argument. Defaults to 0.5.
        alpha (float): 'similar_cut initializer' argument. Must be larger than 1.0. Defaults to 3.0.
        radius (float): 'sculley L1 projection' argument. Defaults to 10.0.
        epsilon (float): 'sculley L1 projection' argument. Defaults to 5.0.
        minimum_df_factor (float): 'minimum df L1 projection' argument. Must be a real number between (0, 1).
        Defaults to 0.01.

    Attributes:
        cluster_centers_ (ndarray): Coordinates of cluster centers, of shape [n_clusters, n_features].
        labels_ (List[int]): Labels of each point.
        inertia_ (float): Sum of squared distances of samples to their closest cluster center.

    Notes:
        The k-means problem is solved using Lloyd's algorithm. The algorithm's complexity is O(k n T), where n is the number of samples and T is the number of iterations. The k-means algorithm is fast but may fall into local minima.
    """
    def __init__(self, n_clusters: int = 8,
                 init: str = 'similar_cut',
                 sparsity: bool = None,
                 max_iter: int = 10,
                 tol: float = 0.0001,
                 verbose: bool = False,
                 random_state: int = 2024,
                 debug_directory: str = None,
                 algorithm: str = None,
                 max_similar: float = 0.5,
                 alpha: float = 3,
                 radius: float = 10.0,
                 epsilon: float = 5.0,
                 minimum_df_factor: float = 0.01):

        self.cluster_centers_ = None
        self.n_clusters = n_clusters
        self.init = init
        self.sparsity = sparsity
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.debug_directory = debug_directory
        self.algorithm = algorithm

        # Similar-cut initialization.
        self.max_similar = max_similar
        self.alpha = alpha

        # Sculley L1 projection.
        self.radius = radius
        self.epsilon = epsilon

        # Minimum df L1 projection.
        self.minimum_df_factor = minimum_df_factor

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k verify input data x is sparse matrix
        """
        X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32])
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))
        if not sp.issparse(X):
            raise ValueError(
                "Input must be instance of scipy.sparse.csr_matrix")
        return X

    def fit(self, X: csr_matrix):
        """Compute k-means clustering.

        Args:
            X : csr_matrix, shape=(n_samples, n_features) Training instances
        """
        X = self._check_fit_data(X)
        self.cluster_centers_, self.labels_, self.inertia_ = (
            k_means(
                X, n_clusters=self.n_clusters, init=self.init,
                sparsity=self.sparsity, max_iter=self.max_iter,
                verbose=self.verbose, tol=self.tol, random_state=self.random_state,
                debug_directory=self.debug_directory, algorithm=self.algorithm,
                max_similar=self.max_similar, alpha=self.alpha, radius=self.radius,
                epsilon=self.epsilon, minimum_df_factor=self.minimum_df_factor
            ))
        return self

    def fit_predict(self, X: csr_matrix) -> ndarray:
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling `fit(X)` followed by `predict(X)`.

        Args:
            X : csr_matrix, shape = (n_samples, n_features).
                New data to be assigned to the closest cluster.

        Returns:
            labels : array, shape = (n_samples,).
                     Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels_

    def transform(self, X: csr_matrix) -> ndarray:
        """Transform X to a cluster-distance space.
        In the new space, each dimension is the distance to the cluster centers.
        Note that even if X is sparse, the array returned by `transform` will typically be dense.

        Parameters:
            X : scipy.sparse.csr_matrix of shape = (n_samples, n_features).
                New data to be transformed to cluster center-distance matrix.

        Returns:
            D : numpy.ndarray
                shape = (n_samples, k)
                D[doc_idx, cluster_idx] = distance(doc_idx, cluster_idx)
        """
        if not hasattr(self, 'cluster_centers_'):
            raise RuntimeError(
                '`transform` function needs centroid vectors. Train SphericalKMeans first.')

        X = self._check_fit_data(X)
        return self._transform(X)

    def _transform(self, X: csr_matrix) -> ndarray:
        """guts of transform method; no input validation"""
        return cosine_distances(X, self.cluster_centers_)


def _tolerance(X, tol):
    """The minimum number of points which are re-assigned to other cluster."""
    return max(1, int(X.shape[0] * tol))


def k_means(X: object, n_clusters: object, init: object = 'similar_cut', sparsity: object = None, max_iter: object = 10,
            verbose: object = False, tol: object = 1e-4, random_state: object = None, debug_directory: object = None,
            algorithm: object = None, max_similar: object = 0.5, alpha: object = 3, radius: object = 10.0,
            epsilon: object = 5.0, minimum_df_factor: object = 0.01) -> object:

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)

    X = as_float_array(X)
    tol = _tolerance(X, tol)

    labels, inertia, centers, debug_header = None, None, None, None

    if debug_directory:
        # Create debug header
        strf_now = datetime.datetime.now()
        debug_header = str(strf_now).replace(
            ':', '-').replace(' ', '_').split('.')[0]

        # Check debug_directory
        if not os.path.exists(debug_directory):
            os.makedirs(debug_directory)

    # For a single thread, run a k-means once
    centers, labels, inertia, n_iter_ = kmeans_single(
        X, n_clusters, max_iter=max_iter, init=init, sparsity=sparsity,
        verbose=verbose, tol=tol, random_state=random_state,
        debug_directory=debug_directory, debug_header=debug_header,
        algorithm=algorithm, max_similar=max_similar, alpha=alpha,
        radius=radius, epsilon=epsilon, minimum_df_factor=minimum_df_factor)

    return centers, labels, inertia


def initialize(X, n_clusters, init, random_state, max_similar, alpha):
    n_samples = X.shape[0]

    # Random selection
    if isinstance(init, str) and init == 'random':
        np.random.seed(random_state)
        seeds = random_state.permutation(n_samples)[:n_clusters]
        centers = X[seeds, :].todense()
    # Customized initial centroids
    elif hasattr(init, '__array__'):
        centers = np.array(init, dtype=X.dtype)
        if centers.shape[0] != n_clusters:
            raise ValueError('the number of customized initial points '
                             'should be same with n_clusters parameter'
                             )
    elif callable(init):
        centers = init(X, n_clusters, random_state=random_state)
        centers = np.asarray(centers, dtype=X.dtype)
    elif isinstance(init, str) and init == 'k-means++':
        centers = _k_init(X, n_clusters, random_state)
    elif isinstance(init, str) and init == 'similar_cut':
        centers = _similar_cut_init(
            X, n_clusters, random_state, max_similar, alpha)
    else:
        raise ValueError('the init parameter for spherical k-means should be '
                         'random, ndarray, k-means++ or similar_cut'
                         )
    centers = normalize(centers)
    return centers


def _k_init(X, n_clusters, random_state):
    """Init n_clusters seeds according to k-means++
    It modified for Spherical k-means

    Parameters
    -----------
    X : sparse matrix, shape (n_samples, n_features)
    n_clusters : integer
        The number of seeds to choose
    random_state : numpy.RandomState
        The generator used to initialize the centers.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007
    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """

    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    random_state = check_random_state(random_state)

    # Set the number of local seeding trials if none is given
    # This is what Arthur/Vassilvitskii tried, but did not report
    # specific results for other than mentioning in the conclusion
    # that it helped.

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    centers[0] = X[center_id].toarray()

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = cosine_distances(centers[0, np.newaxis], X)[0] ** 2
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample() * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)

        centers[c] = X[candidate_ids].toarray()

        # Compute distances to center candidates
        new_dist_sq = cosine_distances(X[candidate_ids, :], X)[0] ** 2
        closest_dist_sq = np.minimum(new_dist_sq, closest_dist_sq)
        current_pot = closest_dist_sq.sum()

    return centers


def _similar_cut_init(X, n_clusters, random_state, max_similar=0.5, sample_factor=3):

    n_data, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    np.random.seed(random_state)
    n_subsamples = min(n_data, int(sample_factor * n_clusters))
    permutation = np.random.permutation(n_data)
    X_sub = X[permutation[:n_subsamples]]
    n_samples = X_sub.shape[0]

    # Pick first center randomly
    center_id = np.random.randint(n_samples)
    center_set = {center_id}
    centers[0] = X[center_id].toarray()
    candidates = np.asarray(range(n_samples))

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        closest_dist = inner_product(
            X_sub[center_id, :], X_sub[candidates, :].T)

        # Remove center similar points from candidates
        remains = np.where(closest_dist.todense() < max_similar)[1]
        if len(remains) == 0:
            break

        np.random.shuffle(remains)
        center_id = candidates[remains[0]]

        centers[c] = X_sub[center_id].toarray()
        candidates = candidates[remains[1:]]
        center_set.add(center_id)

    # If not enough center point search, random sample n_clusters - c points
    n_requires = n_clusters - 1 - c
    if n_requires > 0:
        if n_requires < (n_data - n_subsamples):
            random_centers = permutation[n_subsamples:n_subsamples+n_requires]
        else:
            center_set = set(permutation[np.asarray(list(center_set))])
            random_centers = []
            for idx in np.random.permutation(n_samples):
                if idx in center_set:
                    continue
                random_centers.append(idx)
                if len(random_centers) == n_requires:
                    break

        for i, center_id in enumerate(random_centers):
            centers[c+i+1] = X[center_id].toarray()

    return centers


def kmeans_single(X, n_clusters, max_iter=10, init='similar_cult', sparsity=None,
                  verbose=False, tol=1, random_state=None, debug_directory=None,
                  debug_header=None, algorithm=None, max_similar=0.5, alpha=3,
                  radius=10.0, epsilon=5.0, minimum_df_factor=0.01):

    _initialize_time = time.time()
    centers = initialize(X, n_clusters, init, random_state, max_similar, alpha)
    _initialize_time = time.time() - _initialize_time

    degree_of_sparsity = None
    degree_of_sparsity = check_sparsity(centers)
    ds_strf = ', sparsity={:.3}'.format(
        degree_of_sparsity) if degree_of_sparsity is not None else ''
    initial_state = 'initialization_time={} sec{}'.format(
        '%f' % _initialize_time, ds_strf)

    if verbose:
        print(initial_state)

    if debug_directory:
        log_path = '{}/{}_logs.txt'.format(debug_directory, debug_header)
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('{}\n'.format(initial_state))

    centers, labels, inertia, n_iter_ = _kmeans_single_banilla(
        X, sparsity, n_clusters, centers, max_iter, verbose,
        tol, debug_directory, debug_header,
        radius, epsilon, minimum_df_factor)

    return centers, labels, inertia, n_iter_


def _kmeans_single_banilla(X, sparsity, n_clusters, centers, max_iter,
                           verbose, tol, debug_directory, debug_header,
                           radius, epsilon, minimum_df_factor):

    n_samples = X.shape[0]
    labels_old = np.zeros((n_samples,), dtype=int)

    for n_iter_ in range(1, max_iter + 1):

        _iter_time = time.time()

        labels, distances = pairwise_distances_argmin_min(
            X, centers, metric='cosine')
        centers = _update(X, labels, distances, n_clusters)
        inertia = distances.sum()

        if n_iter_ == 0:
            n_diff = n_samples
        else:
            diff = np.where((labels_old - labels) != 0)[0]
            n_diff = len(diff)

        labels_old = labels

        if isinstance(sparsity, str) and sparsity == 'sculley':
            centers = _sculley_projections(centers, radius, epsilon)
        elif isinstance(sparsity, str) and sparsity == 'minimum_df':
            centers = _minimum_df_projections(
                X, centers, labels_old, minimum_df_factor)

        _iter_time = time.time() - _iter_time

        degree_of_sparsity = None
        degree_of_sparsity = check_sparsity(centers)
        ds_strf = ', sparsity={:.3}'.format(
            degree_of_sparsity) if degree_of_sparsity is not None else ''
        state = 'n_iter={}, changed={}, inertia={}, iter_time={} sec{}'.format(
            n_iter_, n_diff, '%.3f' % inertia, '%.3f' % _iter_time, ds_strf)

        if debug_directory:
            # Log message
            log_path = '{}/{}_logs.txt'.format(debug_directory, debug_header)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write('{}\n'.format(state))

            # Temporal labels
            label_path = '{}/{}_label_iter{}.txt'.format(
                debug_directory, debug_header, n_iter_)
            with open(label_path, 'a', encoding='utf-8') as f:
                for label in labels:
                    f.write('{}\n'.format(label))

            # Temporal cluster_centroid
            center_path = '{}/{}_centroids_iter{}.csv'.format(
                debug_directory, debug_header, n_iter_)
            np.savetxt(center_path, centers)

        if verbose:
            print(state)

        if n_diff <= tol:
            if verbose and (n_iter_ + 1 < max_iter):
                print('Early converged.')
            break

    return centers, labels, inertia, n_iter_


def _update(X, labels, distances, n_clusters):

    n_featuers = X.shape[1]
    centers = np.zeros((n_clusters, n_featuers))

    n_samples_in_cluster = np.bincount(labels, minlength=n_clusters)
    empty_clusters = np.where(n_samples_in_cluster == 0)[0]
    n_empty_clusters = empty_clusters.shape[0]

    data = X.data
    indices = X.indices
    indptr = X.indptr

    if n_empty_clusters > 0:
        # find points to reassign empty clusters to
        far_from_centers = distances.argsort()[::-1][:n_empty_clusters]

        # reassign labels to empty clusters
        for i in range(n_empty_clusters):
            centers[empty_clusters[i]] = X[far_from_centers[i]].toarray()
            n_samples_in_cluster[empty_clusters[i]] = 1
            labels[far_from_centers[i]] = empty_clusters[i]

    # cumulate centroid vector
    for i, curr_label in enumerate(labels):
        for ind in range(indptr[i], indptr[i + 1]):
            j = indices[ind]
            centers[curr_label, j] += data[ind]

    # L2 normalization
    centers = normalize(centers)
    return centers


def _sculley_projections(centers, radius, epsilon):
    n_clusters = centers.shape[0]
    for c in range(n_clusters):
        centers[c] = _sculley_projection(centers[c], radius, epsilon)
    centers = normalize(centers)
    return centers


def _sculley_projection(center, radius, epsilon):
    def l1_norm(x):
        return abs(x).sum()

    def inf_norm(x):
        return abs(x).max()

    upper, lower = inf_norm(center), 0
    current = l1_norm(center)

    larger_than = radius * (1 + epsilon)
    smaller_than = radius

    _n_iter = 0
    theta = 0

    while current > larger_than or current < smaller_than:
        theta = (upper + lower) / 2.0  # Get L1 value for this theta
        current = sum([v for v in (abs(center) - theta) if v > 0])
        if current <= radius:
            upper = theta
        else:
            lower = theta

        # for safety, preventing infinite loops
        _n_iter += 1
        if _n_iter > 10000:
            break
        if upper - lower < 0.001:
            break

    signs = np.sign(center)
    projection = [max(0, ci) for ci in (abs(center) - theta)]
    projection = np.asarray(
        [ci * signs[i] if ci > 0 else 0 for i, ci in enumerate(projection)])
    return projection


def L1_projection(v, z):
    m = v.copy()
    m.sort()
    m = m[::-1]

    pho = 0
    for j, mj in enumerate(m):
        t = mj - (m[:j + 1].sum() - z) / (1 + j)
        if t < 0:
            break
        pho = j

    theta = (m[:pho + 1].sum() - z) / (pho + 1)
    v_ = np.asarray([max(vi - theta, 0) for vi in v])
    return v_


def _minimum_df_projections(X, centers, labels_, minimum_df_factor):
    n_clusters = centers.shape[0]
    centers_ = sp.csr_matrix(centers.copy())

    data = centers_.data
    indptr = centers_.indptr

    n_samples_in_cluster = np.bincount(labels_, minlength=n_clusters)
    min_value = np.asarray([(minimum_df_factor / n_samples_in_cluster[c])
                            if n_samples_in_cluster[c] > 1 else 0 for c in range(n_clusters)])
    for c in range(n_clusters):
        for ind in range(indptr[c], indptr[c + 1]):
            if data[ind] ** 2 < min_value[c]:
                data[ind] = 0
    centers_ = centers_.todense()
    centers_ = normalize(centers_)
    return centers_


def _minimum_df_projection(center, min_value):
    center[[idx for idx, v in enumerate(center) if v**2 < min_value]] = 0
    return center
