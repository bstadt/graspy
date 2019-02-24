# omni.py
# Created by Jaewon Chung on 2018-09-10.
# Email: j1c@jhu.edu
# Copyright (c) 2018. All rights reserved.
import warnings

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..utils import get_lcc, import_graph, is_fully_connected, is_symmetric
from .base import BaseEmbed
from .svd import selectSVD, select_dimension


def _check_valid_graphs(graphs):
    """
    Checks if all graphs in list have same shapes.

    Raises an ValueError if there are more than one shape in the input list,
    or if the list is empty or has one element.

    Parameters
    ----------
    graphs : list
        List of array-like with shapes (n_vertices, n_vertices).

    Raises
    ------
    ValueError
        If all graphs do not have same shape, or input list is empty or has 
        one element.
    """
    if len(graphs) <= 1:
        msg = "Joint RDPG embedding requires more than one graph."
        raise ValueError(msg)

    shapes = set(map(np.shape, graphs))

    if len(shapes) > 1:
        msg = "There are {} different sizes of graphs.".format(len(shapes))
        raise ValueError(msg)

    for graph in graphs:
        if not is_symmetric(graph):
            msg = "Input graphs must be symmetric."
            raise ValueError(msg)


class JointRDPG(BaseEmbed):
    r"""
    Joint random dot product graphs (JRDPG) embeds arbitrary number of input 
    graphs with matched vertex sets.

    Given :math:`A_1, A_2, ..., A_m` a collection of (possibly weighted) adjacency 
    matrices of a collection :math:`m` undirected graphs with matched vertices. 
    Then the :math:`(mn \times mn)` omnibus matrix, :math:`M`, has the subgraph where 
    :math:`M_{ij} = \frac{1}{2}(A_i + A_j)`. The omnibus matrix is then embedded
    using adjacency spectral embedding [1]_.

    Parameters
    ----------
    n_components : int or None, default = None
        Desired dimensionality of output data. If "full", 
        n_components must be <= min(X.shape). Otherwise, n_components must be
        < min(X.shape). If None, then optimal dimensions will be chosen by
        -----.
    n_elbows : int, optional, default: 2
        If `n_compoents=None`, then compute the optimal embedding dimension using
        `select_dimension`. Otherwise, ignored.
    unscaled : bool, optional, default: True
        Whether to scale invidivual eigenvectors with eigenvalues
    algorithm : {'full', 'truncated' (default), 'randomized'}, optional
        SVD solver to use:

        - 'full'
            Computes full svd using ``scipy.linalg.svd``
        - 'truncated'
            Computes truncated svd using ``scipy.sparse.linalg.svd``
        - 'randomized'
            Computes randomized svd using 
            ``sklearn.utils.extmath.randomized_svd``
    n_iter : int, optional (default = 5)
        Number of iterations for randomized SVD solver. Not used by 'full' or 
        'truncated'. The default is larger than the default in randomized_svd 
        to handle sparse matrices that may have large slowly decaying spectrum.
    check_lcc : bool , optional (defult = True)
        Whether to check if the average of all input graphs are connected. May result
        in non-optimal results if the average graph is unconnected. If True and average
        graph is unconnected, a UserWarning is thrown. 

    Attributes
    ----------
    n_graphs_ : int
        Number of graphs
    n_vertices_ : int
        Number of vertices in each graph
    latent_left_ : array, shape (n_samples, n_components)
        Estimated left latent positions of the graph. 
    latent_right_ : array, shape (n_samples, n_components), or None
        Only computed when the graph is directed, or adjacency matrix is 
        asymmetric. Estimated right latent positions of the graph. Otherwise, 
        None.
    singular_values_ : array, shape (n_components)
        Singular values associated with the latent position matrices.
    indices_ : array, or None
        If ``lcc`` is True, these are the indices of the vertices that were 
        kept.

    See Also
    --------
    graspy.embed.selectSVD
    graspy.embed.select_dimension

    References
    ----------
    .. [1] Levin, K., Athreya, A., Tang, M., Lyzinski, V., & Priebe, C. E. (2017, 
       November).A central limit theorem for an omnibus embedding of multiple random 
       dot product graphs. In Data Mining Workshops (ICDMW), 2017 IEEE International 
       Conference on (pp. 964-967). IEEE.
    """

    def __init__(
        self,
        n_components=None,
        n_elbows=2,
        unscaled=True,
        algorithm="randomized",
        n_iter=5,
        check_lcc=True,
    ):
        super().__init__(
            n_components=n_components,
            n_elbows=n_elbows,
            algorithm=algorithm,
            n_iter=n_iter,
            check_lcc=check_lcc,
        )
        self.unscaled = unscaled

    def _reduce_dim(self, graphs):
        # first embed into log2(n_vertices) for each graph
        n_components = int(np.ceil(np.log2(np.min(self.n_vertices_))))

        # embed individual graphs
        Us = []
        Ds = []
        for graph in graphs:
            U, D, _ = selectSVD(
                graph,
                n_components=n_components,
                algorithm=self.algorithm,
                n_iter=self.n_iter,
            )
            Us.append(U)
            Ds.append(D)

        # Choose the best embedding dimension for each graphs
        if self.n_components is None:
            embedding_dimensions = []
            for D in Ds:
                elbows, _ = select_dimension(D, n_elbows=self.n_elbows)
                embedding_dimensions.append(elbows[-1])

            # Choose the max of all of best embedding dimension of all graphs
            best_dimension = np.max(embedding_dimensions)
        else:
            best_dimension = self.n_components

        if self.unscaled:
            Us = np.hstack([U[:, :best_dimension] for U in Us])
        else:
            Us = [U[:, :best_dimension] for U in Us]
            Ds = [D[:best_dimension] for D in Ds]
            Us = np.hstack([U @ np.diag(D) for U, D in zip(Us, Ds)])

        Vhat, _, _ = selectSVD(
            Us,
            n_components=self.n_components,
            n_elbows=self.n_elbows,
            algorithm=self.algorithm,
            n_iter=self.n_iter,
        )

        return Vhat

    def fit(self, graphs, y=None):
        """
        Fit the model with graphs.

        Parameters
        ----------
        graphs : list of graphs, or array-like
            List of array-like, (n_vertices, n_vertices), or list of 
            networkx.Graph. If array-like, the shape must be 
            (n_graphs, n_vertices, n_vertices)
        
        y : Ignored

        Returns
        -------
        self : returns an instance of self.
        """
        # Convert input to np.arrays
        graphs = [import_graph(g) for g in graphs]

        # Check if the input is valid
        _check_valid_graphs(graphs)

        # Save attributes
        self.n_graphs_ = len(graphs)
        self.n_vertices_ = graphs[0].shape[0]

        graphs = np.stack(graphs)

        # Check if Abar is connected
        """if self.check_lcc:
            if not is_fully_connected(graphs.mean(axis=0)):
                msg = (
                    "Input graphs are not fully connected. Results may not"
                    + "be optimal. You can compute the largest connected component by"
                    + "using ``graspy.utils.get_multigraph_union_lcc``."
                )
                warnings.warn(msg, UserWarning)"""

        # embed
        Vhat = self._reduce_dim(graphs)
        self.latent_left_ = Vhat
        self.latent_right_ = None

        return self

    def fit_transform(self, graphs, y=None):
        """
        Fit the model with graphs and apply the embedding on graphs. 
        n_dimension is either automatically determined or based on user input.

        Parameters
        ----------
        graphs : list of graphs
            List of array-like, (n_vertices, n_vertices), or list of 
            networkx.Graph.

        y : Ignored

        Returns
        -------
        out : array-like, shape (n_vertices * n_graphs, n_dimension) if input 
            graphs were symmetric. If graphs were directed, returns tuple of 
            two arrays (same shape as above) where the first corresponds to the
            left latent positions, and the right to the right latent positions
        """
        return self._fit_transform(graphs)
