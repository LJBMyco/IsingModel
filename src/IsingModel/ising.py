"""Create the model."""

import math
from typing import Literal
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
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
        dynamics: Literal["glauber", "kawasaki"],
        temperature: float,
        energy_j: Optional[int] = 1,
        k_b: Optional[float] = 1.0,
        rng: Optional[Generator] = None,
    ):
        """Initialise a lattice.

        Args:
            shape (Tuple[int,int]): MxN shape to make the lattice.
            dynamics (Literal[glauber, kawasaki]): Dynamics to run the model.
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

        self.dynamics = dynamics
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

    def energy_at_site(self, i_index: int, j_index: int) -> float:
        """Energy at a given site.

        Args:
            i_index (int): Position along first axis
            j_index (int): Position along second axis

        Returns:
            float: Total energy at site.
        """
        site_energy = 0.0
        site_energy += self.lattice[pbc(self.shape[0], i_index - 1)][j_index]
        site_energy += self.lattice[pbc(self.shape[0], i_index + 1)][j_index]
        site_energy += self.lattice[i_index][pbc(self.shape[1], j_index + 1)]
        site_energy += self.lattice[i_index][pbc(self.shape[1], j_index - 1)]

        return self.energy_j * site_energy * self.lattice[i_index, j_index]

    def glauber_update(self) -> None:
        """Use Glauber dynamics to update the lattice."""
        i_index = self.rng.integers(0, self.shape[0])
        j_index = self.rng.integers(0, self.shape[1])

        delta_energy = self.glauber_energy(i_index, j_index)

        swap = self.metropolis_test(delta_energy)

        if swap:
            self.lattice[i_index][j_index] *= -1.0

    def glauber_energy(self, i_index: int, j_index: int) -> float:
        """Calculate the energy change by swapping the flip at given site.

        Args:
            i_index (int): Position along first axis
            j_index (int): Position along second axis

        Returns:
            float: Change in energy
        """
        return 2 * self.energy_at_site(i_index, j_index)

    def kawasaki_update(self):
        """Use Kawasaki Dynamics to update the model."""
        i1 = 0
        i2 = 0
        j1 = 0
        j2 = 0

        while (i1 == i2) and (j1 == j2):
            i1 = self.rng.integers(0, self.shape[0])
            i2 = self.rng.integers(0, self.shape[0])
            j1 = self.rng.integers(0, self.shape[1])
            j2 = self.rng.integers(0, self.shape[1])

        if self.lattice[i1][j1] != self.lattice[i2][j2]:
            delta_energy = self.kawasaki_energy(i1, i2, j1, j2)
            swap = self.metropolis_test(delta_energy)
            if swap:
                self.lattice[i1][j1] *= -1
                self.lattice[i2][j2] *= -1

    def kawasaki_energy(self, i1: int, i2: int, j1: int, j2: int) -> float:
        """Calculate the energy change by swapping the flip at two sites.

        Args:
            i1 (int): Position along first axis of site 1.
            i2 (int): Position along first axis of site 2.
            j1 (int): Position along second axis of site 1.
            j2 (int): Position along second axis of site 2.

        Returns:
            float: Energy change
        """
        site_1_swap = 2.0 * self.energy_at_site(i1, j1)
        site_2_swap = 2.0 * self.energy_at_site(i2, j2)
        total_energy = site_1_swap + site_2_swap
        if (i1 == i2) and (
            j1 in [pbc(j2 + 1, self.shape[1]), pbc(j2 - 1, self.shape[1])]
        ):
            total_energy -= 2.0 * self.energy_j
        if (j1 == j2) and (
            i1 in [pbc(i2 + 1, self.shape[0]), pbc(i2 - 1, self.shape[0])]
        ):
            total_energy -= 2.0 * self.energy_j

        return site_1_swap + site_2_swap

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

    def update(self):
        """Animation update."""
        for _ in range(self.shape[0] * self.shape[1]):
            if self.dynamics == "glauber":
                self.glauber_update()
            elif self.dynamics == "kawasaki":
                self.kawasaki_update()

    def frame_update(self, i):
        """Animation frame update."""
        self.update()
        self.mat.set_data(self.lattice)
        self.text.set_text(f"Sweep: {i+1}/{self.animation_frames}")
        return (self.mat,)

    def animate(self, frames: Optional[int] = None):
        """Animate model."""
        fig, ax = plt.subplots()
        self.animation_frames = frames
        self.mat = ax.imshow(self.lattice)
        self.text = ax.text(
            s=f"Sweep: 0/{frames}",
            x=0.3,
            y=1.01,
            transform=ax.transAxes,
            fontsize="xx-large",
        )
        if frames is not None:
            ani = FuncAnimation(
                fig,
                self.frame_update,
                interval=1,
                blit=False,
                repeat=False,
                cache_frame_data=False,
                frames=frames,
            )
        else:
            ani = FuncAnimation(
                fig,
                self.frame_update,
                interval=1,
                blit=False,
                repeat=False,
                cache_frame_data=False,
            )

        return ani
