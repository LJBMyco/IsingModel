"""Create the model."""

from typing import Optional
from typing import Tuple

import numpy as np
from numpy.random import Generator


class Model:
    """Class to contain the model."""

    def __init__(
        self,
        shape: Tuple[int, int],
        energy_j: Optional[int] = 1,
        rng: Optional[Generator] = None,
    ):
        """Initialise a lattice.

        Args:
            shape (Tuple[int,int]): MxN shape to make the lattice.
            energy_j (int, optional): Amout energy of system is lowered by aligned pair.
                Defaults to 1.
            rng (Generator, optional.): Numpy random number Generator.
                Defaults to np.random.default_rng().
        """
        self.rng = rng or np.random.default_rng()

        self.shape = shape

        self.lattice = self.rng.random(self.shape)
        self.lattice[self.lattice >= 0.5] = 1.0
        self.lattice[self.lattice != 1.0] = -1.0

        self.energy_j = energy_j

    @property
    def magnetism(self):
        """Calculate total magnesitm of the lattice."""
        return np.abs(np.sum(self.lattice))

    @property
    def energy(self):
        """Calculate total energy in the lattice."""
        total_energy_lattice = self.lattice * (
            np.roll(self.lattice, 1, 0) + np.roll(self.lattice, 1, 1)
        )

        return -self.energy_j * total_energy_lattice.sum()
