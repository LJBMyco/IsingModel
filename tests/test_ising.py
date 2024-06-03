"""Test the ising module."""

import numpy as np


def test_model():
    """Test the Model class."""
    from IsingModel.ising import Model

    shape = (10, 5)
    model = Model(shape=shape)

    assert model.shape[0] == 10
    assert model.shape[1] == 5
    assert not np.any((-1.0 < model.lattice) & (model.lattice < 1.0))
    assert not np.any(model.lattice > 1.0)
    assert not np.any(model.lattice < -1.0)


def test_total_magnetism():
    """Test total magnetism is correctly calculated."""
    from IsingModel.ising import Model

    shape = (5, 5)
    model = Model(shape=shape)

    model.lattice = np.ones(model.shape)
    assert model.magnetism == 25

    model.lattice = -np.ones(model.shape)
    assert model.magnetism == 25

    model.lattice = np.zeros(model.shape)
    assert model.magnetism == 0
