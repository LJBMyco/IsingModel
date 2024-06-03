"""Create the model."""

from typing import Optional
from typing import Tuple

import numpy as np
from numpy.random import Generator


class Model:
    """Class to contain the model."""

    def __init__(self, shape: Tuple[int, int], rng: Optional[Generator] = None):
        """Initialise a lattice.

        Args:
            shape (Tuple[int,int]): MxN shape to make the lattice.
            rng (Generator, optional.): Numpy random number Generator.
                Defaults to np.random.default_rng().
        """
        self.rng = rng or np.random.default_rng()

        self.shape = shape
        self.lattice = self.rng.random(self.shape)
        self.lattice[self.lattice >= 0.5] = 1.0
        self.lattice[self.lattice != 1.0] = -1.0

    @property
    def magnetism(self):
        """Calculate total magnesitm of the lattice."""
        return np.abs(np.sum(self.lattice))
