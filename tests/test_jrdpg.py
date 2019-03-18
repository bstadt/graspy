import numpy as np
import pytest
from numpy import allclose, array_equal
from numpy.linalg import norm
from numpy.testing import assert_allclose

from graspy.cluster.gclust import GaussianCluster
from graspy.embed.jrdpg import JointRDPG
from graspy.simulations.simulations import er_np, sbm
from graspy.utils.utils import is_symmetric, symmetrize


def make_train(n=[20, 20], m=10, directed=False):
    """
    Make 4 class training dataset
    n = number of vertices
    m = number of graphs from each class
    """
    p1 = [[0.5, 0.2], [0.2, 0.5]]
    p2 = [[0.2, 0.5], [0.5, 0.2]]
    p3 = [[0.2, 0.15], [0.15, 0.2]]
    A = [sbm(n, p) for _ in range(m) for p in [p1, p2, p3]]

    if directed:
        p4 = [[0.25, 0.1], [0.5, 0.9]]  # Assymetric
        A += [sbm(n, p4, directed=True) for _ in range(m)]
    else:
        p4 = [[0.05, 0.25], [0.25, 0.05]]
        A += [sbm(n, p4) for _ in range(m)]

    return A


def test_bad_inputs():
    with pytest.raises(TypeError):
        "Invalid unscaled"
        unscaled = "1"
        jrdpg = JointRDPG(unscaled=unscaled)

    with pytest.raises(ValueError):
        "Test single graph input"
        np.random.seed(1)
        A = er_np(100, 0.2)
        JointRDPG().fit(A)

    with pytest.raises(ValueError):
        "Test graphs with different sizes"
        np.random.seed(1)
        A = [er_np(100, 0.2)] + [er_np(200, 0.2)]
        JointRDPG().fit(A)


def test_graph_clustering():
    """
    There should be 4 total clusters since 4 class problem.
    n_components = 2
    """
    # undirected case
    np.random.seed(2)
    n = [20, 20]
    m = 10
    X = make_train(n, m)

    res = JointRDPG(n_components=2).fit(X).scores_.reshape((m * 4, -1))
    gmm = GaussianCluster(max_components=10).fit(res)

    assert gmm.n_components_ == 4

    # directed case
    """np.random.seed(3)
    X = make_train(n, m, directed=True)

    res = JointRDPG(n_components=2).fit(X).scores_.reshape((m * 4, -1))
    gmm = GaussianCluster(max_components=10).fit(res)

    assert gmm.n_components_ == 4"""


def test_vertex():
    """
    There should be 2 clusters since all SBMs are 2-block model.
    """
    # undirected case
    np.random.seed(4)
    n = [20, 20]
    m = 10
    X = make_train(n, m)

    res = JointRDPG(n_components=2).fit(X).latent_left_
    gmm = GaussianCluster(max_components=10).fit(res)

    assert gmm.n_components_ == 2

    # directed case
    """np.random.seed(5)
    X = make_train(n, m, directed=True)

    jrdpg = JointRDPG(n_components=2).fit(X)
    res = np.hstack([jrdpg.latent_left_, jrdpg.latent_right_])
    gmm = GaussianCluster(max_components=10).fit(res)"""
