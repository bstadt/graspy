import pytest
import numpy as np
from numpy import array_equal, allclose
from numpy.testing import assert_allclose
from numpy.linalg import norm

from graspy.embed.jrdpg import JointRDPG
from graspy.simulations.simulations import er_np, er_nm
from graspy.utils.utils import symmetrize, is_symmetric


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


def test_undirected_inputs():
    pass
