"""Create the model."""

import math
from typing import Optional
from typing import Tuple

import numpy as np
from numpy.random import Generator


def pbc(shape: int, index: int) -> int:
    """Periodic boundary conditions of a lattice.

    Args:
        shape (int): Size of axis.
        index (int): Index on axis.

    Returns:
        int: Index with pbc applied.
    """
    if (index > shape - 1) or (index < 0):
        index = index % shape
    return index


class Model:
    """Class to contain the model."""

    def __init__(
        self,
        shape: Tuple[int, int],
        temperature: float,
        energy_j: Optional[int] = 1,
        k_b: Optional[float] = 1.0,
        rng: Optional[Generator] = None,
    ):
        """Initialise a lattice.

        Args:
            shape (Tuple[int,int]): MxN shape to make the lattice.
            temperature (float): Temperature of the system.
            energy_j (int, optional): Amout energy of system is lowered by aligned pair.
                Defaults to 1.
            k_b (float, optional): Boltzmann constant. Defaults to 1.0.
            rng (Generator, optional.): Numpy random number Generator.
                Defaults to np.random.default_rng().
        """
        self.rng = rng or np.random.default_rng()

        self.shape = shape

        self.lattice = self.rng.random(self.shape)
        self.lattice[self.lattice >= 0.5] = 1.0
        self.lattice[self.lattice != 1.0] = -1.0

        self.temperature = temperature
        self.k_b = k_b

        self.base_exponential = math.exp(-1.0 / (self.k_b * self.temperature))

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

    def glauber_update(self) -> None:
        """Use Glauber dynamics to update the lattice."""
        i_index = self.rng.integers(0, self.shape[0])
        j_index = self.rng.integers(0, self.shape[1])

        delta_energy = self.glauber_energy(i_index, j_index)

        swap = self.metropolis_test(delta_energy)

        if swap:
            self.lattice[i_index][j_index] *= 1.0

    def glauber_energy(self, i_index: int, j_index: int) -> float:
        """Calculate the energy change by swapping the flip at given site.

        Args:
            i_index (int): Position along first axis
            j_index (int): Position along second axis

        Returns:
            float: Change in energy
        """
        swap_energy = 0.0
        swap_energy += self.lattice[pbc(self.shape[0], i_index - 1)][j_index]
        swap_energy += self.lattice[pbc(self.shape[0], i_index + 1)][j_index]
        swap_energy += self.lattice[i_index][pbc(self.shape[1], j_index + 1)]
        swap_energy += self.lattice[i_index][pbc(self.shape[1], j_index - 1)]

        return 2.0 * self.energy_j * swap_energy * self.lattice[i_index, j_index]

    def metropolis_test(self, delta_energy: float) -> bool:
        """Determine if a flip should be completed.

        Args:
            delta_energy (float): Change in energy.

        Returns:
            bool: True if keep change. False is not.
        """
        if delta_energy <= 0.0:
            return True

        else:
            random_number = self.rng.random()
            probability = self.base_exponential**delta_energy
            if random_number <= probability:
                return True

        return False
