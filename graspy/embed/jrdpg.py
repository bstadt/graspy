import warnings

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..utils import get_lcc, import_graph, is_fully_connected, is_almost_symmetric
from .base import BaseEmbed, _check_valid_graphs
from .svd import select_dimension, selectSVD


class JointRDPG(BaseEmbed):
    r"""
    Joint random dot product graphs (JRDPG) embeds arbitrary number of input 
    graphs with matched vertex sets.

    Given :math:`A_1, A_2, ..., A_m` a collection of (possibly weighted) adjacency 
    matrices of a collection :math:`m`

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
        Whether to scale invidivual eigenvectors with eigenvalues in first embedding 
        stage.
    algorithm : {'full', 'truncated', 'randomized' (default)}, optional
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

    Attributes
    ----------
    n_graphs_ : int
        Number of graphs
    n_vertices_ : int
        Number of vertices in each graph
    latent_left_ : array, shape (n_samples, n_components)
        Estimated left latent positions of the graph. 
    latent_right_ : array, shape (n_samples, n_components), or None
        Estimated right latent positions of the graph. Only computed when the an input 
        graph is directed, or adjacency matrix is assymetric. Otherwise, None.
    scores_ : array, shape (n_samples, n_components, n_components)

    Notes
    -----
    When an input graph is directed, `n_components` of `latent_left_` may not be equal to
    `n_components` of `latent_right_`.
    """

    def __init__(
        self,
        n_components=None,
        n_elbows=2,
        unscaled=True,
        algorithm="randomized",
        n_iter=5,
    ):
        super().__init__(
            n_components=n_components,
            n_elbows=n_elbows,
            algorithm=algorithm,
            n_iter=n_iter,
        )
        self.unscaled = unscaled

    def _reduce_dim(self, graphs):
        # first embed into log2(n_vertices) for each graph
        n_components = int(np.ceil(np.log2(np.min(self.n_vertices_))))

        # embed individual graphs
        embeddings = [
            selectSVD(
                graph,
                n_components=n_components,
                algorithm=self.algorithm,
                n_iter=self.n_iter,
            )
            for graph in graphs
        ]
        Us, Ds, Vs = zip(*embeddings)

        # Choose the best embedding dimension for each graphs
        if self.n_components is None:
            embedding_dimensions = []
            for D in Ds:
                elbows, _ = select_dimension(D, n_elbows=self.n_elbows)
                embedding_dimensions.append(elbows[-1])

            # Choose the median of all of best embedding dimension of all graphs
            best_dimension = int(np.ceil(np.median(embedding_dimensions)))
        else:
            best_dimension = self.n_components

        if self.unscaled:
            Us = np.hstack([U[:, :best_dimension] for U in Us])
            Vs = np.hstack([V.T[:, :best_dimension] for V in Vs])
        else:
            Us = np.hstack(
                [
                    U[:, :best_dimension] @ np.diag(D[:best_dimension])
                    for U, D in zip(Us, Ds)
                ]
            )
            Vs = np.hstack(
                [
                    V[:, :best_dimension] @ np.diag(V[:best_dimension])
                    for V, D in zip(Vs, Ds)
                ]
            )

        # Second SVD for vertices
        # The notation is slightly different than the paper
        Uhat, _, _ = selectSVD(
            Us,
            n_components=self.n_components,
            n_elbows=self.n_elbows,
            algorithm=self.algorithm,
            n_iter=self.n_iter,
        )

        Vhat, _, _ = selectSVD(
            Vs,
            n_components=self.n_components,
            n_elbows=self.n_elbows,
            algorithm=self.algorithm,
            n_iter=self.n_iter,
        )
        return Uhat, Vhat

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

        # Check if undirected
        undirected = any(is_almost_symmetric(g) for g in graphs)

        # Save attributes
        self.n_graphs_ = len(graphs)
        self.n_vertices_ = graphs[0].shape[0]

        graphs = np.stack(graphs)

        # embed
        Uhat, Vhat = self._reduce_dim(graphs)
        self.latent_left_ = Uhat
        if not undirected:
            self.latent_right_ = Vhat
            self.scores_ = Uhat.T @ graphs @ Vhat
        else:
            self.latent_right_ = None
            self.scores_ = Uhat.T @ graphs @ Uhat

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
