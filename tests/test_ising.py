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


def test_total_energy():
    """Test total energy of the system."""
    from IsingModel.ising import Model

    shape = (5, 5)
    energy_j = 2

    model = Model(shape=shape, energy_J=energy_j)
    model.lattice = np.array([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]])
    assert model.energy == 12

    model.lattice = np.ones(shape)
    assert model.energy == -100

    model.lattice *= -1.0
    assert model.energy == -100


def test_pbc():
    """Test periodic boundary conditions."""
    from IsingModel.ising import pbc

    assert pbc(5, 1) == 1
    assert pbc(5, 6) == 1
    assert pbc(5, -1) == 4


def test_metroplis_test():
    """Test the metroplis test."""
    from IsingModel.ising import Model

    model = Model(shape=(5, 5), temperature=1.0)

    assert model.metropolis_test(0.0)
    assert model.metropolis_test(-1.0)


def test_glauber_energy():
    """Test calculating delta E using glauber."""
    from IsingModel.ising import Model

    model = Model(shape=(3, 3), temperature=1.0)

    model.lattice = np.array([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]])

    assert model.glauber_energy(1, 1) == -8
    assert model.glauber_energy(0, 1) == -4
