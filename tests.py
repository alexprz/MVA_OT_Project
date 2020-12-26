"""Implement tests."""
import numpy as np
import unittest
from hypothesis import strategies as strats
from hypothesis import given

from sparse_deconvolution_1D import paper_env


@strats.composite
def env(draw):
    seed = draw(strats.integers(0, 2**32 - 1))
    np.random.seed(seed)
    return paper_env(5)


class TestSparseDeconvolution(unittest.TestCase):
    """Test the sparse deconvolution example."""

    @given(env=env())
    def test_w(self, env):
        assert env.w.shape == (5,)
        abs_w = np.absolute(env.w)
        assert ((0.5 <= abs_w) & (abs_w <= 1.5)).all()

    @given(env=env())
    def test_p(self, env):
        assert env.p.shape == (5,)
        assert ((0 <= env.p) & (env.p <= 1)).all()

    @given(env=env())
    def test_y(self, env):
        env.y(np.linspace(0, 1, 10))

    @given(env=env())
    def test_R(self, env):
        r1 = env.R(np.ones(10))
        r2 = env.R(np.ones((1, 10)))
        r3 = env.R(np.ones((2, 10)))

        assert r1.shape == (1, )
        assert r2.shape == (1,)
        assert r3.shape == (2,)

    @given(env=env())
    def test_phi(self, env):
        m, N = 10, 5
        f = env.phi(w=np.arange(m), theta=np.ones(m), x=np.linspace(0, 1, N))
        assert f.shape == (m, N)

        m, N, d = 10, 5, 1
        f = env.phi(w=np.arange(m), theta=np.ones((m, d)),
                    x=np.linspace(0, 1, N))
        assert f.shape == (m, N)
        f = env.phi(w=np.ones((m, 1)), theta=np.ones((m, d)),
                    x=np.linspace(0, 1, N))
        assert f.shape == (m, N)

    @given(env=env())
    def test_grad_R(self, env):
        w = np.arange(10)
        theta = np.arange(10)
        grad_w, grad_theta = env.grad_R(w, theta, np.ones(10))

        assert grad_w.shape == w.shape
        assert grad_theta.shape == theta.shape

        w = np.ones((10, 1))
        theta = np.ones((10, 1))
        grad_w, grad_theta = env.grad_R(w, theta, np.ones(10))

        assert grad_w.shape == (w.shape[0],)
        assert grad_theta.shape == (theta.shape[0],)
