import numpy as np

from ase.neighborlist import (
    primitive_neighbor_list,
    first_neighbors,
    NewPrimitiveNeighborList,
)


class PrimitiveNeighborListWrapper(NewPrimitiveNeighborList):
    """Slightly modified ASE NewPrimitiveNeighborList class, such that it can return
    vertor distances as well
    """

    def build(self, pbc, cell, positions, numbers=None):
        """Build the list."""
        self.pbc = np.array(pbc, copy=True)
        self.cell = np.array(cell, copy=True)
        self.positions = np.array(positions, copy=True)

        self.pair_first, self.pair_second, self.offset_vec, self.dist_vec = (
            primitive_neighbor_list(
                "ijSD",
                pbc,
                cell,
                positions,
                self.cutoffs,
                numbers=numbers,
                self_interaction=self.self_interaction,
                use_scaled_positions=self.use_scaled_positions,
            )
        )

        if len(positions) > 0 and not self.bothways:
            mask = np.logical_or(
                np.logical_and(
                    self.pair_first <= self.pair_second,
                    (self.offset_vec == 0).all(axis=1),
                ),
                np.logical_or(
                    self.offset_vec[:, 0] > 0,
                    np.logical_and(
                        self.offset_vec[:, 0] == 0,
                        np.logical_or(
                            self.offset_vec[:, 1] > 0,
                            np.logical_and(
                                self.offset_vec[:, 1] == 0, self.offset_vec[:, 2] > 0
                            ),
                        ),
                    ),
                ),
            )
            self.pair_first = self.pair_first[mask]
            self.pair_second = self.pair_second[mask]
            self.offset_vec = self.offset_vec[mask]

        if len(positions) > 0 and self.sorted:
            mask = np.argsort(self.pair_first * len(self.pair_first) + self.pair_second)
            self.pair_first = self.pair_first[mask]
            self.pair_second = self.pair_second[mask]
            self.offset_vec = self.offset_vec[mask]

        # Compute the index array point to the first neighbor
        self.first_neigh = first_neighbors(len(positions), self.pair_first)

        self.nupdates += 1

    def get_neighbors(self, a):
        return (
            self.pair_second[self.first_neigh[a] : self.first_neigh[a + 1]],
            self.offset_vec[self.first_neigh[a] : self.first_neigh[a + 1]],
            self.dist_vec[self.first_neigh[a] : self.first_neigh[a + 1]],
        )
