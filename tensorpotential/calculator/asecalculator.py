from __future__ import annotations

import logging
import math
import numpy as np
import time
import bisect
from collections import deque
from dataclasses import dataclass

from tensorflow.data import Dataset

from typing import Any, Dict, Tuple, Optional, List, NamedTuple
from itertools import combinations_with_replacement
from ase.calculators.calculator import Calculator, all_changes

from tensorpotential.tensorpot import TensorPotential
from tensorpotential.tpmodel import (
    extract_cutoff_and_elements,
    extract_cutoff_matrix,
    load_model_metadata,
)
# uq.constants is imported lazily inside the few call sites — eager
# import would deadlock because uq/__init__.py imports feature_extraction
# which imports back from this module (TPCalculator).
from tensorpotential.data.databuilder import (
    GeometricalDataBuilder,
    get_number_of_real_atoms,
    get_number_of_real_neigh,
    AbstractDataBuilder,
)
from tensorpotential import constants

log = logging.getLogger(__name__)


@dataclass
class AdaptivePaddingConfig:
    """Tuning knobs for adaptive miss-rate-driven padding growth.

    Used by PaddingManager when ``adaptive_padding=True``.
    """

    miss_rate_window: int = 10
    miss_rate_threshold: float = 0.5
    atoms_growth_factor: float = 2.0
    neighbors_growth_factor: float = 1.5
    max_pad_atoms_number: int = 50
    max_pad_neighbors_fraction: float = 0.50

    def __post_init__(self):
        if self.atoms_growth_factor <= 1.0:
            raise ValueError(
                f"atoms_growth_factor must be > 1.0, but got {self.atoms_growth_factor}"
            )
        if self.neighbors_growth_factor <= 1.0:
            raise ValueError(
                f"neighbors_growth_factor must be > 1.0, but got {self.neighbors_growth_factor}"
            )


class PaddingManager:
    """
    Manages padding for batches of atomic structures to ensure consistent input sizes for models.

    This class handles padding based on either a fraction of neighbors or a fixed number of atoms,
    or a combination of both. It maintains a history of padding bounds to reuse existing bounds
    if possible and to potentially reduce padding if it becomes excessive.

    Args:
        data_builders (List[DataBuilderProtocol]): A list of DataBuilder objects that handle
            data extraction, batching, and padding for different aspects of the atomic structures.
        pad_neighbors_fraction (Optional[float]):  Fraction of real neighbors to add as padding.
            Must be between 0 and 1 inclusive if provided. Defaults to 0.10. If None, neighbor padding
            is not based on a fraction.
        pad_atoms_number (Optional[int]): Number of atoms to add as padding. Must be a positive integer
            if provided. Defaults to 10. If None, atom padding is not based on a fixed number.
        max_number_reduction_recompilation (Optional[int]): Maximum number of times padding can be
            reduced and recompilation triggered. If None, padding reduction is disabled. Defaults to 3.
        debug_padding_verbose (int): Verbosity level for padding debug output. Defaults to 0.
            0 — silent, 1 — adaptive padding events only (margin growth),
            2 — any padding bounds changes (new bounds, reductions),
            3 — all hits/misses and per-call details.
    """

    def __init__(
        self,
        data_builders: List[AbstractDataBuilder],
        pad_neighbors_fraction: Optional[float] = 0.10,
        pad_atoms_number: Optional[int] = 10,
        max_number_reduction_recompilation: Optional[int] = 3,
        debug_padding_verbose: int = 0,
        adaptive_padding: bool = True,
        adaptive_padding_config: Optional[AdaptivePaddingConfig] = None,
        # deprecated — use debug_padding_verbose instead
        debug_padding: Optional[bool] = None,
    ):
        if pad_neighbors_fraction is not None and not 0 < pad_neighbors_fraction <= 1:
            raise ValueError(
                f"pad_neighbors_fraction must be a fraction between 0 and 1, but got {pad_neighbors_fraction}"
            )
        if pad_atoms_number is not None:
            if not isinstance(pad_atoms_number, int):
                raise TypeError(
                    f"pad_atoms_number must be an integer, but got {type(pad_atoms_number)}"
                )
            if pad_atoms_number <= 0:
                raise ValueError(
                    f"pad_atoms_number must be larger than 0, but got {pad_atoms_number}"
                )

        # Backward compat: debug_padding=True → verbose level 3
        if debug_padding is not None:
            debug_padding_verbose = 3 if debug_padding else 0

        if adaptive_padding_config is None:
            adaptive_padding_config = AdaptivePaddingConfig()

        self.data_builders: List[AbstractDataBuilder] = data_builders
        self.pad_neighbors_fraction: Optional[float] = pad_neighbors_fraction
        self.pad_atoms_number: Optional[int] = pad_atoms_number
        self.max_number_reduction_recompilation: Optional[int] = (
            max_number_reduction_recompilation
        )
        self.number_reduction_recompilation: int = 0
        self.debug_padding_verbose: int = debug_padding_verbose
        self.padding_fraction_history = deque(maxlen=10_000)

        # List of tuples (max_atoms, max_neighbors), kept sorted via bisect.insort
        self.padding_bounds: List[Tuple[int, int]] = []

        # Adaptive padding state
        self.adaptive_padding: bool = adaptive_padding
        self.adaptive_padding_config: AdaptivePaddingConfig = adaptive_padding_config

        self._call_history: deque = deque(maxlen=adaptive_padding_config.miss_rate_window)
        self._current_pad_atoms_number: Optional[int] = pad_atoms_number
        self._current_pad_neighbors_fraction: Optional[float] = pad_neighbors_fraction
        self._n_adaptations: int = 0

    def find_upper_padding_bound(
        self, max_nat: int, max_nneigh: int
    ) -> Optional[Tuple[int, int]]:
        """
        Finds the smallest existing padding bound in history that is greater than or equal to
        the current required bounds (max_nat, max_nneigh).

        ``padding_bounds`` is a list of ``(atoms, neighbors)`` tuples kept in
        lexicographic order by ``bisect.insort``. That ordering is monotone in
        the atoms dimension but *not* in the neighbors dimension (different
        atom counts can have arbitrarily ordered neighbor counts), so a true
        2-D binary search isn't sound. The lookup is a hybrid:
          * O(log n) binary search to skip all entries with atoms < max_nat;
          * O(k) linear scan over the tail to find the first entry whose
            neighbor count also satisfies the requirement.
        In the typical regime n is small (<= a few dozen bounds) so the tail
        scan is cheap; correctness, not asymptotic cost, drives the shape.

        Args:
            max_nat (int): The maximum number of atoms in the current batch.
            max_nneigh (int): The maximum number of neighbors in the current batch.

        Returns:
            Optional[Tuple[int, int]]: An existing upper padding bound (max_atoms, max_neighbors)
                                        from history if found, otherwise None.
        """
        # O(log n): skip all entries with atoms < max_nat
        idx = bisect.bisect_left(self.padding_bounds, (max_nat,))
        # Scan only entries with atoms >= max_nat for neighbors match
        for i in range(idx, len(self.padding_bounds)):
            if self.padding_bounds[i][1] >= max_nneigh:
                return self.padding_bounds[i]
        return None

    def get_padded_bound(self, nreal_atoms: int, nreal_neigh: int) -> Tuple[int, int]:
        """
        Calculates the padded bounds for atoms and neighbors based on the configured padding parameters.

        When adaptive_padding is enabled, uses the current (possibly grown) margins
        instead of the original fixed values.

        Args:
            nreal_atoms (int): The number of real atoms in the current structure.
            nreal_neigh (int): The number of real neighbors in the current structure.

        Returns:
            Tuple[int, int]: The calculated padded bounds as a tuple (max_atoms, max_neighbors).
        """
        pad_neigh_frac = self._current_pad_neighbors_fraction
        pad_atoms_num = self._current_pad_atoms_number

        current_max_neighbors: int
        if pad_neigh_frac is not None:
            current_max_neighbors = max(
                256,
                math.ceil(nreal_neigh + nreal_neigh * pad_neigh_frac),
            )
        else:
            current_max_neighbors = nreal_neigh

        current_max_atoms: int
        if pad_atoms_num is not None:
            current_max_atoms = int(nreal_atoms + pad_atoms_num)
        else:
            current_max_atoms = nreal_atoms

        # Atoms are cheap relative to bonds — ensure atom count is at least
        # as large as any existing bound with fewer neighbors, so the new
        # bound dominates those bounds and covers more future queries.
        for existing in self.padding_bounds:
            if existing[1] <= current_max_neighbors and existing[0] > current_max_atoms:
                current_max_atoms = existing[0]

        return current_max_atoms, current_max_neighbors

    def _record_and_maybe_adapt(self, is_miss: bool):
        """Record a cache hit/miss and grow padding margins if miss rate is too high."""
        if not self.adaptive_padding:
            return
        cfg = self.adaptive_padding_config
        self._call_history.append(is_miss)
        if len(self._call_history) < cfg.miss_rate_window:
            return
        miss_rate = sum(self._call_history) / len(self._call_history)
        if miss_rate <= cfg.miss_rate_threshold:
            return

        old_atoms = self._current_pad_atoms_number
        old_neigh = self._current_pad_neighbors_fraction

        self._current_pad_atoms_number = min(
            math.ceil(self._current_pad_atoms_number * cfg.atoms_growth_factor),
            cfg.max_pad_atoms_number,
        )
        self._current_pad_neighbors_fraction = min(
            self._current_pad_neighbors_fraction * cfg.neighbors_growth_factor,
            cfg.max_pad_neighbors_fraction,
        )

        self._n_adaptations += 1
        self._call_history.clear()

        if self._n_adaptations == 1:
            # First growth: surface at INFO so users running with default
            # logging see that adaptive padding has kicked in (subsequent
            # growths stay at debug verbosity to avoid log spam).
            log.info(
                "Adaptive padding grew margins for the first time "
                "(miss_rate=%.2f > %.2f): pad_atoms %d -> %d, "
                "pad_neighbors_frac %.4f -> %.4f. Further growths are "
                "silent unless debug_padding_verbose >= 1.",
                miss_rate, cfg.miss_rate_threshold,
                old_atoms, self._current_pad_atoms_number,
                old_neigh, self._current_pad_neighbors_fraction,
            )

        if self.debug_padding_verbose >= 1:
            print(
                f"Adaptive padding: miss_rate={miss_rate:.2f} > {cfg.miss_rate_threshold}, "
                f"growing margins: pad_atoms {old_atoms} -> {self._current_pad_atoms_number}, "
                f"pad_neighbors_frac {old_neigh:.4f} -> {self._current_pad_neighbors_fraction:.4f} "
                f"(adaptation #{self._n_adaptations})"
            )

    def get_data(self, atoms) -> Dict[str, Any]:
        """
        Processes the input atomic structure (atoms) to extract data, batch it, and apply padding.

        This method orchestrates the data processing pipeline:
        1. Extracts data features from the atomic structure using DataBuilders.
        2. Joins the extracted data into a batch format.
        3. Determines and applies padding to the batch based on the current padding strategy
           and history.

        Args:
            atoms: An atomic structure object (e.g., ASE Atoms object).

        Returns:
            Dict[str, Any]: The processed and padded batch of data.
        """
        # Stage 1: Data Extraction
        data_dict: Dict[str, Any] = {}
        for data_builder in self.data_builders:
            data_dict.update(data_builder.extract_from_ase_atoms(atoms))

        # Stage 2: Batching
        batch_group: List[Dict[str, Any]] = [
            data_dict
        ]  # Consider if batch_group is always of size 1
        batch: Dict[str, Any] = {}
        for data_builder in self.data_builders:
            batch.update(data_builder.join_to_batch(batch_group))

        # Stage 3: Determine Real Number of Atoms and Neighbors
        nreal_atoms: int = get_number_of_real_atoms(batch)
        nreal_neigh: int = get_number_of_real_neigh(batch)

        if self.debug_padding_verbose >= 3:
            print(f"nreal_atoms={nreal_atoms}, nreal_neigh={nreal_neigh}")

        # Stage 4: Padding
        if self.pad_atoms_number and self.pad_neighbors_fraction:
            if not self.padding_bounds:
                # FIRST TIME - NO current_max_neighbors and current_max_atoms, setup with padded threshold
                upper_bound = self.get_padded_bound(nreal_atoms, nreal_neigh)
                # insert into ordered list
                if self.debug_padding_verbose >= 2:
                    print(f"Adding initial padding bound: {upper_bound}")
                bisect.insort(self.padding_bounds, upper_bound)
                self._record_and_maybe_adapt(is_miss=True)
            else:
                # extract padding bounds from self.padding_bounds
                upper_bound: Optional[Tuple[int, int]] = self.find_upper_padding_bound(
                    nreal_atoms, nreal_neigh
                )
                if self.debug_padding_verbose >= 3:
                    print(f"extract padding bounds: {upper_bound}")
                if upper_bound is None:
                    upper_bound = self.get_padded_bound(nreal_atoms, nreal_neigh)
                    # insert into ordered list
                    if self.debug_padding_verbose >= 2:
                        print(f"Adding new maximum padding bound: {upper_bound}")
                    # if upper_bound not in self.padding_bounds:
                    bisect.insort(self.padding_bounds, upper_bound)
                    self._record_and_maybe_adapt(is_miss=True)
                else:
                    self._record_and_maybe_adapt(is_miss=False)
                    # check if too large padding
                    if self.is_large_padding(nreal_atoms, nreal_neigh, upper_bound):
                        if self.max_number_reduction_recompilation is not None and (
                            self.number_reduction_recompilation
                            < self.max_number_reduction_recompilation
                        ):
                            self.number_reduction_recompilation += 1
                            upper_bound = self.get_padded_bound(
                                nreal_atoms, nreal_neigh
                            )
                            # insert into ordered list
                            if self.debug_padding_verbose >= 2:
                                print(
                                    f"Reducing padding and adding new bound: {upper_bound}"
                                )
                            # if upper_bound not in self.padding_bounds:
                            bisect.insort(self.padding_bounds, upper_bound)
                        elif self.debug_padding_verbose >= 3:
                            print(
                                f"Padding reduction skipped or max reductions reached. Keeping bound: {upper_bound}"
                            )
                    elif self.debug_padding_verbose >= 3:
                        print(f"Using existing padding bound: {upper_bound}")

            pad_max_n_atoms, pad_max_n_neighbors = upper_bound

            batch_max_pad_dict: Dict[str, int] = {
                constants.PAD_MAX_N_STRUCTURES: 1,
                constants.PAD_MAX_N_ATOMS: pad_max_n_atoms,
                constants.PAD_MAX_N_NEIGHBORS: pad_max_n_neighbors,
            }

            padding_fraction = (pad_max_n_atoms + pad_max_n_neighbors) / (
                nreal_atoms + nreal_neigh
            ) - 1
            if self.debug_padding_verbose >= 3:
                print("padding_fraction={:.2f}".format(padding_fraction))
                self.padding_fraction_history.append(padding_fraction)

            for data_builder in self.data_builders:
                data_builder.pad_batch(batch, batch_max_pad_dict)

        return batch

    def is_large_padding(self, nreal_atoms, nreal_neigh, upper_bound):
        npad_atoms = upper_bound[0] - nreal_atoms
        npad_neigh = upper_bound[1] - nreal_neigh
        padding_fraction = (npad_atoms + npad_neigh) / (nreal_atoms + nreal_neigh)
        return (
            npad_atoms + npad_neigh > 1000
        ) and padding_fraction > self.pad_neighbors_fraction


def _snap_up_hybrid(x: int, step: int = 16, linear_max: int = 128,
                    geom_factor: Optional[float] = 1.5) -> int:
    """Round ``x`` UP onto a hybrid ladder: a fine linear grid (multiples of ``step``) up to
    ``linear_max``, then a geometric grid (×``geom_factor``) above. Keeps padding tight in the
    common low range while bounding the number of distinct rungs in the heavy tail (so a diverse
    sequence collapses onto ~log-many widths instead of one shape per structure).
    ``geom_factor=None`` disables the geometric tail entirely → pure linear (the "tight" regime)."""
    if x <= 0:
        return step
    if geom_factor is None or x <= linear_max:
        return int(math.ceil(x / step) * step)
    r = linear_max
    while r < x:
        r = int(math.ceil(r * geom_factor))
    return r


def _ema(prev: Optional[float], x: float, alpha: float) -> float:
    """Exponential moving average update; seeds with ``x`` on the first sample (``prev is None``)."""
    return x if prev is None else (1.0 - alpha) * prev + alpha * x


class DensePaddingManager:
    """Padding manager for the dense (reshape) calculator: sequential single-structure evaluation
    of (possibly very) heterogeneous structures, minimizing XLA recompiles first and padded data
    second. The dense compute cost is ``atoms·width`` (every padded slot runs the full per-bond
    pipeline), so reuse-vs-recompile is decided on that cost — unlike the segment_sum
    ``PaddingManager`` which trades on ``atoms+neighbors``.

    Strategy (mirrors the seg manager, adapted to the dense ``(atoms, width)`` shape):
      * keep the set of visited ``(atoms, width)`` shapes (== compiled graphs);
      * for a structure ``(nat, true_max)``, reuse the smallest visited shape that fits
        (``A≥nat, W≥true_max``) if its dense waste ``A·W/(nat·true_max)−1`` ≤ ``reuse_tolerance``;
      * otherwise mint a new shape snapped onto the hybrid ladder (one compile), bounding the
        number of *optional* (tightening) mints by ``max_extra_shapes``; a structure larger than
        every visited shape always mints (mandatory), but minted shapes live on the ladder so the
        total compile count is bounded by the ladder size.
      * hot-shape promotion (measured ski-rental): a coarse shape reused by a stable, narrow
        region (MD/relaxation, reused thousands of times) is replaced by a tight shape sized to
        the observed envelope (exact atoms, tightened width) -- but only once the padding it
        wastes has accumulated to one measured XLA-compile cost (calibrated via ``note_eval``).
        Self-calibrating per device; never loses on short runs; diverse scans never trigger it
        (wide envelope -> no gain).
    """

    def __init__(
        self,
        data_builders,
        reuse_tolerance: float = 0.25,
        max_extra_shapes: int = 64,
        width_floor: int = 64,
        width_step: int = 16,
        width_linear_max: int = 128,
        width_geom_factor: Optional[float] = 1.5,
        atom_floor: int = 16,
        atom_step: int = 16,
        atom_linear_max: int = 128,
        atom_geom_factor: Optional[float] = 1.5,
        tight: bool = False,
        promote: bool = True,
        promote_safety: float = 2.0,
        promote_margin: float = 0.12,
        promote_step: int = 8,
        promote_min_gain: float = 0.05,
        ema_alpha: float = 0.3,
        verbose: int = 0,
    ):
        # "tight" regime: pure-linear small-step snapping (no floor64, no geometric ladder) -- pad
        # close to each structure's true shape from the start. For a single structure / MD /
        # relaxation / uniform dataset (calculator mode="uniform"), as opposed to the coarse
        # shape-collapsing ladder used for a diverse scan.
        if tight:
            width_floor = atom_floor = width_step = atom_step = 8
            width_geom_factor = atom_geom_factor = None
        self.data_builders = data_builders
        self.reuse_tolerance = reuse_tolerance
        self.max_extra_shapes = max_extra_shapes
        self.width_floor, self.atom_floor = width_floor, atom_floor
        self.width_step, self.width_linear_max, self.width_geom_factor = (
            width_step, width_linear_max, width_geom_factor)
        self.atom_step, self.atom_linear_max, self.atom_geom_factor = (
            atom_step, atom_linear_max, atom_geom_factor)
        # Hot-shape promotion (measured ski-rental): a coarse shape reused by a STABLE
        # (narrow-envelope) region is an MD/relaxation workload, where the shape is reused
        # thousands of times -> the optimal padding margin tends to 0. We mint a tight replacement
        # (atoms == max nat seen, exact for fixed-N; width tightened, capped at the coarse width)
        # only once the padding wasted by reusing the coarse shape has accumulated to the cost of
        # one XLA compile -- both measured at runtime via ``note_eval`` (the first eval of a shape
        # carries its compile; later evals are steady). This is self-calibrating per device and,
        # unlike a fixed reuse count, never loses on short runs (a run too short to pay back a
        # compile never reaches the threshold -> never promotes). ``promote=False`` disables it.
        # The per-step saving estimate assumes eval cost ~ padded slots (an upper bound, since some
        # cost is fixed); ``promote_safety`` (>= 1) discounts for that -> raise it to be more
        # conservative. The gain guard makes promotion a no-op for diverse-collapse (wide envelope
        # -> tight ~ coarse, no gain).
        self.promote = promote
        self.promote_safety = promote_safety
        self.promote_margin = promote_margin
        self.promote_step = promote_step
        self.promote_min_gain = promote_min_gain
        self.ema_alpha = ema_alpha
        self.verbose = verbose
        self.visited: List[Tuple[int, int]] = []   # (atoms, width), kept sorted
        self._extra_mints: int = 0
        self.n_compiles: int = 0
        self.n_reuse: int = 0
        self.n_promote: int = 0
        # per visited shape: the (max nat, max true_max) envelope of the structures that reused it
        self._env: Dict[Tuple[int, int], Tuple[int, int]] = {}
        # measured-cost state (calibrated by note_eval): shape of the last selection, EMA compile
        # time (s), per-shape + global EMA steady eval time (s), per-shape eval counter (1st ==
        # compile), and per coarse shape the accumulated estimated saving (s) vs its tight twin.
        self._last_shape: Optional[Tuple[int, int]] = None
        self.c_compile_s: Optional[float] = None
        self._eval_s: Dict[Tuple[int, int], float] = {}
        self._eval_s_global: Optional[float] = None
        self._eval_seen: Dict[Tuple[int, int], int] = {}
        self._saved_s: Dict[Tuple[int, int], float] = {}

    def _snap_width(self, w: int) -> int:
        # width floor collapses all low-coordination structures onto one width (analogous to the
        # seg manager's min-256 neighbor floor) -- the main lever for matching seg's compile count.
        return max(self.width_floor,
                   _snap_up_hybrid(w, self.width_step, self.width_linear_max, self.width_geom_factor))

    def _snap_atoms(self, n: int) -> int:
        return max(self.atom_floor,
                   _snap_up_hybrid(n, self.atom_step, self.atom_linear_max, self.atom_geom_factor))

    def _record(self, shape: Tuple[int, int]) -> None:
        if shape not in self.visited:
            bisect.insort(self.visited, shape)
            self.n_compiles += 1
            if self.verbose >= 1:
                log.info("dense calc: new shape %s -> compile #%d (visited=%d)",
                         shape, self.n_compiles, len(self.visited))

    def _tight_promote(self, shape: Tuple[int, int]) -> Tuple[int, int]:
        """Tight replacement for a hot coarse ``shape``, sized to its observed envelope. Atoms are
        set to the max nat seen (exact for fixed-N MD/relaxation); width is tightened toward the
        envelope max (plus a small drift margin) but never above the coarse width (which already
        fits), so the result is never larger than ``shape`` -- the gain comes from whichever
        dimension the coarse shape padded (atoms for high-coordination, width for floor-padded
        low-coordination)."""
        max_nat, max_tm = self._env[shape]
        w = int(math.ceil(max_tm * (1.0 + self.promote_margin) / self.promote_step)
                * self.promote_step)
        w = max(self.promote_step, min(shape[1], w))
        return (max_nat, w)

    def _maybe_promote(self, best: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Measured ski-rental: accumulate the per-step padding saving a tight twin of ``best``
        would yield, and promote (mint it) once that has paid back one measured compile. Returns
        the promoted shape, or None to keep reusing ``best``."""
        if not self.promote or self._extra_mints >= self.max_extra_shapes:
            return None
        # Not yet calibrated (need a compile sample and a steady-eval sample). Checked first so the
        # common steady-state reuse path skips the envelope sizing until measurements exist.
        eval_s = self._eval_s.get(best, self._eval_s_global)
        if eval_s is None or self.c_compile_s is None:
            return None
        tight = self._tight_promote(best)
        best_slots, tight_slots = best[0] * best[1], tight[0] * tight[1]
        # no real gain (e.g. diverse-collapse: wide envelope -> tight ~ coarse) -> never promote
        if tight in self.visited or tight_slots > best_slots * (1.0 - self.promote_min_gain):
            return None
        # eval cost ~ padded slots: saving per reuse from running tight instead of coarse
        saved_per_step = eval_s * (best_slots - tight_slots) / best_slots
        self._saved_s[best] = self._saved_s.get(best, 0.0) + saved_per_step
        if self._saved_s[best] < self.c_compile_s * self.promote_safety:
            return None  # the wasted padding has not yet paid back a compile -> keep reusing
        self._extra_mints += 1
        self.n_promote += 1
        self._record(tight)
        self._env[tight] = self._env[best]
        self._saved_s[best] = 0.0
        if self.verbose >= 1:
            log.info("dense calc: ski-rental promote %s -> %s (saved %.2fs >= %.1fx compile %.2fs)",
                     best, tight, self.c_compile_s * self.promote_safety,
                     self.promote_safety, self.c_compile_s)
        return tight

    def select_shape(self, nat: int, true_max: int) -> Tuple[int, int]:
        """Pick the dense ``(atoms, width)`` shape for one structure (pure decision; updates the
        visited set / counters and records the selection for ``note_eval``)."""
        true_max = max(int(true_max), 1)
        real = nat * true_max
        fits = [s for s in self.visited if s[0] >= nat and s[1] >= true_max]
        best = min(fits, key=lambda s: s[0] * s[1]) if fits else None
        if best is None:
            chosen = (self._snap_atoms(nat), self._snap_width(true_max))  # mandatory mint
            self._record(chosen)
            self._env[chosen] = (nat, true_max)
        else:
            chosen = best
            waste = (best[0] * best[1]) / real - 1.0
            if waste > self.reuse_tolerance and self._extra_mints < self.max_extra_shapes:
                tighter = (self._snap_atoms(nat), self._snap_width(true_max))
                if tighter not in self.visited:  # tighter than every fit -> worth a compile
                    self._extra_mints += 1
                    self._record(tighter)
                    self._env[tighter] = (nat, true_max)
                    chosen = tighter
            if chosen is best:
                # reuse: extend the (nat, true_max) envelope of best's users, then the ski-rental test
                self.n_reuse += 1
                e = self._env.get(best, best)
                self._env[best] = (max(e[0], nat), max(e[1], true_max))
                promoted = self._maybe_promote(best)
                if promoted is not None:
                    chosen = promoted
        self._last_shape = chosen
        return chosen

    def note_eval(self, elapsed_s: float) -> None:
        """Calibration feedback from the calculator: wall-time (s) of the model.compute that ran on
        the shape returned by the most recent ``select_shape``. The first eval of a shape carries
        its XLA compile; later evals are steady. Feeds the measured ski-rental promotion. No-op
        until a shape has been selected; safe to call unconditionally."""
        sh = self._last_shape
        if sh is None or elapsed_s <= 0:
            return
        seen = self._eval_seen.get(sh, 0) + 1
        self._eval_seen[sh] = seen
        a = self.ema_alpha
        if seen == 1:  # first eval of this shape == compile spike
            self.c_compile_s = _ema(self.c_compile_s, elapsed_s, a)
        else:  # steady eval -> per-shape and global EMA
            self._eval_s[sh] = _ema(self._eval_s.get(sh), elapsed_s, a)
            self._eval_s_global = _ema(self._eval_s_global, elapsed_s, a)

    def get_data(self, atoms) -> Dict[str, Any]:
        data_dict: Dict[str, Any] = {}
        for db in self.data_builders:
            data_dict.update(db.extract_from_ase_atoms(atoms))
        batch: Dict[str, Any] = {}
        for db in self.data_builders:
            batch.update(db.join_to_batch([data_dict]))

        nat = get_number_of_real_atoms(batch)
        n_real = get_number_of_real_neigh(batch)
        ind_i = np.asarray(batch[constants.BOND_IND_I])[:n_real]
        true_max = int(np.bincount(ind_i, minlength=nat).max()) if nat else 1

        max_atoms, width = self.select_shape(nat, true_max)
        max_pad_dict: Dict[str, int] = {
            constants.PAD_MAX_N_STRUCTURES: 1,
            constants.PAD_MAX_N_ATOMS: max_atoms,
            constants.PAD_MAX_NEIGH: width,
            constants.PAD_MAX_N_NEIGHBORS: max_atoms * width,
        }
        for db in self.data_builders:
            db.pad_batch(batch, max_pad_dict)
        return batch


class _LoadedModel(NamedTuple):
    model: Any
    data_keys: List[str]
    uq_sigs: Dict[str, Any]
    path: Optional[str]
    can_seg: bool   # model can run the segment_sum engine (`compute`)
    can_dense: bool  # model can run the dense reshape engine (`compute_dense`)


def _load_one_model(m) -> "_LoadedModel":
    """Resolve a model spec to a ``_LoadedModel`` (model, data_keys, uq_sigs, path, can_seg, can_dense).

    Engine capability:
      * SavedModels may export BOTH a ``compute`` (segment_sum) and a ``compute_dense``
        (reshape) signature from the same weights -- they can be dual-engine.
      * In-memory TPModels are single-engine, fixed by how they were built (the
        instructions' ``dense_nbr``): a model whose equivariant SPBF was traced with
        ``dense_nbr=True`` runs dense only, otherwise segment_sum only.

    ``can_seg``/``can_dense`` report what THIS model can run; the calculator picks the
    engine from ``mode`` and (for a SavedModel running dense) rebinds ``m.compute`` to
    ``m.compute_dense``. ``compute`` and ``compute_dense`` share an identical input
    signature, so ``data_keys`` is the same either way.

    ``uq_sigs`` maps UQ mode name → ConcreteFunction (empty for in-memory TPModels).
    """
    if isinstance(m, str):
        from tensorpotential.uq import constants as uq_constants
        m_path = m
        m = TensorPotential.load_model(m)
        can_dense = "compute_dense" in m.signatures
        can_seg = "compute" in m.signatures or "serving_default" in m.signatures
        # data_keys are identical across compute/compute_dense; pick whichever exists
        if "compute" in m.signatures:
            sigs = m.signatures["compute"]
        elif "compute_dense" in m.signatures:
            sigs = m.signatures["compute_dense"]
        elif "serving_default" in m.signatures:
            sigs = m.signatures["serving_default"]
        else:
            raise ValueError("Neither `compute` nor `serving_default` found")
        uq_sigs = {
            mode: m.signatures[sig_name]
            for mode, sig_name in uq_constants.UQ_MODE_TO_SIGNATURE.items()
            if sig_name in m.signatures
        }
        return _LoadedModel(m, list(sigs._arg_keywords), uq_sigs, m_path, can_seg, can_dense)
    if hasattr(m, "compute") and callable(m.compute):
        # in-memory TPModel: single-engine, determined by the instructions' dense_nbr
        instr = getattr(m, "instructions", None)
        it_iter = instr.values() if hasattr(instr, "values") else (instr or [])
        is_dense = any(
            getattr(it, "dense_capable", False) and getattr(it, "dense_nbr", False)
            for it in it_iter
        )
        return _LoadedModel(m, list(m.compute_specs.keys()), {}, None, (not is_dense), is_dense)
    raise ValueError("model type is not recognized")


class TPCalculator(Calculator):
    """
    Atomic Simulation Environment (ASE) calculator for TensorPotential models.

    Aggregation engine -- the ``mode`` argument (two regimes; no single engine wins everywhere):

      * ``mode="uniform"`` -- **dense** neighbor aggregation (the reshape ``compute_dense`` path).
        Use it when every evaluated structure has (nearly) the same shape: a SINGLE structure,
        an MD or relaxation trajectory, or a uniform dataset. Dense is ~1.3-1.7x faster per
        structure and is *order-invariant* (its compiled-shape count does not depend on the
        order structures arrive in). It pads TIGHT from the start (see ``DensePaddingManager``),
        so it is the wrong choice for highly heterogeneous inputs, where tight padding forces a
        fresh XLA compile per distinct size.

      * ``mode="diverse"`` (default) -- **segment_sum** aggregation. Use it when evaluating MANY
        very different structures (e.g. scanning a heterogeneous dataset). For such a scan, sort
        the structures by DECREASING atom count: the first (largest) structure's compiled shape
        then covers all the rest, minimizing recompiles (ascending order is the worst case -- it
        can mint a fresh compile per size step, an order of magnitude more than descending).

      If the model cannot run the requested engine (in-memory TPModels are single-engine, fixed by
      how they were built; a SavedModel may export only ``compute``), the calculator falls back to
      the available engine and logs a warning. SavedModels exported from a dense-capable model
      carry both engines and can switch freely.

    Args:
        model (Any): The TensorPotential model. This can be:
            - A path to a tf.saved_model file (string).
            - An instance of a TPModel
        mode (str, optional): ``"uniform"`` (dense) or ``"diverse"`` (segment_sum, default).
            See the engine discussion above.
        pad_neighbors_fraction (float, optional): Fraction by which to extend
            the neighbor list with fake neighbors (between 0 and 1, for XLA compiled models).
        pad_atoms_number (int, optional): number of fake atoms to pad (for XLA compiled models).
        adaptive_padding (bool, optional): Enable miss-rate-driven margin growth (default True).
        adaptive_padding_config (AdaptivePaddingConfig, optional): Tuning knobs for adaptive
            padding (window size, growth factors, caps). Defaults to AdaptivePaddingConfig().
        **kwargs: Additional keyword arguments passed to the base ASE Calculator class.
            Recognized extras: ``cutoff`` (float) — override the cutoff extracted from the
            model.

    """

    implemented_properties = ["energy", "forces", "stress", "free_energy"]

    def __init__(
        self,
        model: list[Any] | Any,
        pad_neighbors_fraction: float | None = 0.05,
        pad_atoms_number: int | None = 1,
        min_dist=None,
        extra_properties: list[str] = None,
        truncate_extras_by_natoms: bool | list[str] = False,
        max_number_reduction_recompilation: int | None = 2,
        debug_padding_verbose: int = 0,
        adaptive_padding: bool = True,
        adaptive_padding_config: AdaptivePaddingConfig | None = None,
        enable_uq_if_available: bool = True,
        mode: str = "diverse",
        # deprecated — use debug_padding_verbose instead
        debug_padding: bool | None = None,
        **kwargs,
    ):
        if model is None:
            raise ValueError('"model" parameter is not provided')
        cutoff = kwargs.pop("cutoff", None)
        Calculator.__init__(self, **kwargs)
        self.data = None
        self.cutoff = cutoff

        self.eval_time = 0
        self.basis = None

        self.min_dist = min_dist  # minimal distance
        self.compute_properties = ["energy", "forces", "free_energy", "stress"]

        self.extra_properties = list(extra_properties) if extra_properties else None
        self.truncate_extras_by_natoms = truncate_extras_by_natoms
        self.gmm_uq_model = None
        # Map of UQ mode → ConcreteFunction; empty when no UQ signatures
        # are available (in-memory TPModel, or SavedModel without UQ).
        self._saved_model_uq_sigs: Dict[str, Any] = {}
        # Set to a snapshot dict when UQ is enabled; None when disabled.
        self._uq_state: Optional[Dict[str, Any]] = None

        # Engine mode (see class docstring): "uniform" -> dense reshape aggregation;
        # "diverse" -> segment_sum. Resolved against each model's capabilities below.
        mode = str(mode).lower()
        if mode not in ("uniform", "diverse"):
            raise ValueError(
                f"mode must be 'uniform' (dense, for a single structure or a uniform "
                f"dataset) or 'diverse' (segment_sum, for many heterogeneous structures), "
                f"got {mode!r}"
            )
        self.mode = mode
        is_ensemble = isinstance(model, list)
        model_specs = model if is_ensemble else [model]
        self.models = []
        self.data_keys = []
        model_paths = []  # track paths for metadata loading
        can_seg, can_dense = True, True   # engine capability ANDed across the ensemble
        for i, m in enumerate(model_specs):
            r = _load_one_model(m)
            can_seg, can_dense = can_seg and r.can_seg, can_dense and r.can_dense
            model_paths.append(r.path)
            self.models.append(r.model)
            if i == 0:
                self.data_keys = r.data_keys
                if not is_ensemble:
                    self._saved_model_uq_sigs = r.uq_sigs
            elif not set(self.data_keys).issubset(r.data_keys):
                raise ValueError(
                    f"Models have inconsistent data keys: expected superset of "
                    f"{self.data_keys}, got {r.data_keys}"
                )

        # Resolve the requested mode against what the model(s) can actually run. Each mode
        # uses its preferred engine when available, else falls back to the other (with a
        # warning) -- some models are single-engine (in-memory TPModels; seg-only SavedModels).
        want_dense = self.mode == "uniform"
        if want_dense and can_dense:
            self.dense_reshape = True
        elif (not want_dense) and can_seg:
            self.dense_reshape = False
        elif can_dense and not can_seg:
            self.dense_reshape = True
            log.warning(
                "mode='diverse' (segment_sum) requested, but this model only provides the "
                "dense engine -- using dense. (Rebuild/export with a segment_sum `compute` "
                "signature for the diverse engine.)"
            )
        elif can_seg and not can_dense:
            self.dense_reshape = False
            log.warning(
                "mode='uniform' (dense) requested, but this model has no `compute_dense` "
                "signature -- falling back to segment_sum. (Export the model with a dense "
                "signature to use the uniform engine.)"
            )
        else:
            raise ValueError("model supports neither a segment_sum nor a dense compute engine")

        # On a SavedModel running dense, rebind the default callable so model.compute(...)
        # dispatches to the fast dense path (shares the same weights; identical inputs).
        if self.dense_reshape:
            for mdl in self.models:
                if not isinstance(mdl, str) and hasattr(mdl, "compute_dense"):
                    mdl.compute = mdl.compute_dense

        if not self.dense_reshape:
            log.info(
                "Calculator in 'diverse' mode (segment_sum): for a sequential scan of many "
                "heterogeneous structures, sort them by DECREASING atom count -- this lets the "
                "first (largest) structure's compiled shape cover the rest and minimizes XLA "
                "recompiles. For a single structure or a uniform dataset use mode='uniform'."
            )

        cutoffs, element_maps, cutoff_matrices = [], [], []
        for i, model in enumerate(self.models):
            # Try metadata.yaml first (reliable for saved models)
            model_path = model_paths[i] if i < len(model_paths) else None
            metadata = load_model_metadata(model_path)
            if metadata is not None and "cutoff" in metadata and "chemical_symbols" in metadata:
                cutoffs.append(metadata["cutoff"])
                symbols = metadata["chemical_symbols"]
                element_maps.append(
                    {sym: idx for idx, sym in enumerate(symbols)}
                )
                if "cutoff_matrix" in metadata:
                    cutoff_matrices.append(np.array(metadata["cutoff_matrix"]))
            else:
                # Fall back to extracting from instruction objects
                cutoff, element_map_symbols, element_map_index = (
                    extract_cutoff_and_elements(model.instructions)
                )
                cutoffs.append(cutoff)
                element_maps.append(
                    {k: v for k, v in zip(element_map_symbols, element_map_index)}
                )
                cutoff_matrix = extract_cutoff_matrix(model.instructions)
                if cutoff_matrix is not None:
                    cutoff_matrices.append(cutoff_matrix)
        cutoff = np.max(cutoffs)
        assert all(
            [ems == element_maps[0]] for ems in element_maps
        )  # check that all maps are identical
        self.element_map = element_maps[0]

        actual_cutoff_matrices = [m for m in cutoff_matrices if m is not None]

        if len(actual_cutoff_matrices) == 0:
            self.cutoff_dict = None
            if cutoff > 0:
                if self.cutoff is not None and self.cutoff != cutoff:
                    print(
                        f"Cutoff of the potential {cutoff} A is different from calculator's {self.cutoff} A. "
                        f"Using the value from the potential."
                    )
                self.cutoff = cutoff
            else:
                print(
                    f"Couldn't extract cutoff value from the model. Using the value from calculator: {self.cutoff}A"
                )
        else:
            assert len(set([m.shape for m in actual_cutoff_matrices])) == 1
            # get one max matrix over possibly many
            nelems = len(self.element_map)
            matrix = np.max(actual_cutoff_matrices, axis=0).reshape(nelems, nelems)
            # get single max value
            self.cutoff = np.max(matrix)
            # construct dict
            all_bond_comb = combinations_with_replacement(np.arange(nelems), 2)
            inv_element_map = {v: k for k, v in self.element_map.items()}

            self.cutoff_dict = {}
            for comb in all_bond_comb:
                el0, el1 = inv_element_map[comb[0]], inv_element_map[comb[1]]
                self.cutoff_dict[(el0, el1)] = matrix[comb[0], comb[1]]

        self.geom_data_builder = GeometricalDataBuilder(
            elements_map=self.element_map,
            cutoff=self.cutoff,
            cutoff_dict=self.cutoff_dict,
            # float_dtype="float64",
            # In dense mode the builder's pad_batch emits the per-atom-uniform reshape layout,
            # driven by the DensePaddingManager-selected (atoms, width) shape.
            dense_nbr=self.dense_reshape,
        )
        self.data_builders = [self.geom_data_builder]
        if constants.ATOMIC_MAGMOM in self.data_keys:
            try:
                from tensorpotential.experimental.mag.databuilder import (
                    MagMomDataBuilder,
                )
            except ModuleNotFoundError:
                raise ImportError(
                    "TensorPotential.experimental.mag.databuilder not found"
                )

            self.data_builders.append(MagMomDataBuilder())
        if constants.ATOMIC_POS in self.data_keys:
            try:
                from tensorpotential.extra.gen_tensor.databuilder import (
                    PositionsDataBuilder,
                )
            except ModuleNotFoundError:
                raise ImportError(
                    "TensorPotential.extra.gen_tensor.databuilder not found"
                )

            self.data_builders.append(PositionsDataBuilder(cutoff=self.cutoff))
        if constants.CELL_VECTORS in self.data_keys:
            try:
                from tensorpotential.extra.gen_tensor.databuilder import (
                    CellDataBuilder,
                )
            except ModuleNotFoundError:
                raise ImportError(
                    "TensorPotential.extra.gen_tensor.databuilder not found"
                )

            self.data_builders.append(CellDataBuilder(cutoff=self.cutoff))

        self.padding_manager = PaddingManager(
            data_builders=self.data_builders,
            pad_neighbors_fraction=pad_neighbors_fraction,
            pad_atoms_number=pad_atoms_number,
            max_number_reduction_recompilation=max_number_reduction_recompilation,
            debug_padding_verbose=debug_padding_verbose,
            adaptive_padding=adaptive_padding,
            adaptive_padding_config=adaptive_padding_config,
            debug_padding=debug_padding,
        )
        # Dense ("uniform") calculator: a manager that selects the per-atom-uniform (atoms, width)
        # shape per structure. In uniform mode the workload is a single structure or a uniform
        # dataset, so we pad TIGHT from the start (tight=True) instead of the coarse shape-collapsing
        # ladder (which only helps a diverse scan, and a diverse scan should use
        # mode='diverse'/segment_sum). The ski-rental promotion stays enabled as a dormant safety
        # net: it only acts if a 'uniform' dataset turns out more spread than expected.
        self.dense_padding_manager = (
            DensePaddingManager(data_builders=self.data_builders, tight=True)
            if self.dense_reshape
            else None
        )
        if self._saved_model_uq_sigs and enable_uq_if_available:
            self.enable_uq()
            log.info(
                "Uncertainty quantification is enabled (mode='%s'). "
                "Call calc.enable_uq(mode='gamma_only'/'full') to switch, "
                "or calc.disable_uq() to use standard compute.",
                self._uq_state["mode"],
            )

    @property
    def _uq_enabled(self) -> bool:
        return self._uq_state is not None

    @property
    def available_uq_modes(self) -> list[str]:
        """List of UQ modes exposed by the loaded SavedModel."""
        return list(self._saved_model_uq_sigs.keys())

    def enable_uq(self, mode: str | None = None):
        """Switch to a compute_uq signature (UQ + standard outputs).

        Only works when the calculator was loaded from a SavedModel that
        carries at least one UQ signature.  ``mode`` is one of
        ``"full"`` (the HAL compute, with ``dsigma/dr`` and
        ``virial_sigma``) or ``"gamma_only"`` (cheaper, scalar
        uncertainty fields only).  Defaults to ``"full"`` when available,
        otherwise ``"gamma_only"``.  Snapshots the original state so
        ``disable_uq()`` can restore it."""
        from tensorpotential.uq import constants as uq_constants

        if not self._saved_model_uq_sigs:
            raise RuntimeError(
                "No compute_uq signature found. The model must be saved with "
                "gmm_uq_model= via save_model_with_aux_computes()."
            )
        if mode is None:
            mode = (
                uq_constants.UQ_MODE_FULL
                if uq_constants.UQ_MODE_FULL in self._saved_model_uq_sigs
                else uq_constants.UQ_MODE_GAMMA_ONLY
            )
        if mode not in self._saved_model_uq_sigs:
            raise ValueError(
                f"UQ mode {mode!r} not available; SavedModel exposes "
                f"{list(self._saved_model_uq_sigs)}"
            )
        if self._uq_enabled and self._uq_state["mode"] == mode:
            return
        if self._uq_enabled:
            # Mode switch: restore baseline first so the snapshot is clean
            self.disable_uq()
        sig = self._saved_model_uq_sigs[mode]
        model = self.models[0]
        self._uq_state = {
            "mode": mode,
            "compute": model.compute,
            "data_keys": list(self.data_keys),
            "extra_properties": (
                list(self.extra_properties)
                if self.extra_properties is not None
                else None
            ),
            "truncate": self.truncate_extras_by_natoms,
        }
        # SavedModel ConcreteFunctions are kwargs-only; the regular TPModel
        # ``compute`` takes a dict positionally. ``model.compute(self.data)``
        # is the call shape downstream, so wrap to splat the dict.
        sig_kwargs = sig._arg_keywords
        def _compute_via_signature(data, _sig=sig, _keys=sig_kwargs):
            return _sig(**{k: data[k] for k in _keys})
        model.compute = _compute_via_signature
        self.data_keys = list(sig_kwargs)
        self._configure_uq_properties_from_signature(sig)

    def disable_uq(self):
        """Restore the standard compute signature, removing UQ outputs."""
        if self._uq_state is None:
            return
        model = self.models[0]
        snapshot = self._uq_state
        model.compute = snapshot["compute"]
        self.data_keys = snapshot["data_keys"]
        self.extra_properties = snapshot["extra_properties"]
        self.truncate_extras_by_natoms = snapshot["truncate"]
        self._uq_state = None

    def _set_uq_extra_properties(self, extra, truncate):
        """Merge UQ keys into self.extra_properties and self.truncate_extras_by_natoms."""
        if self.extra_properties is None:
            self.extra_properties = extra
        else:
            for k in extra:
                if k not in self.extra_properties:
                    self.extra_properties.append(k)
        if isinstance(self.truncate_extras_by_natoms, list):
            for k in truncate:
                if k not in self.truncate_extras_by_natoms:
                    self.truncate_extras_by_natoms.append(k)
        else:
            self.truncate_extras_by_natoms = truncate

    def _configure_uq_properties_from_signature(self, uq_sig):
        """Auto-configure extra_properties from a compute_uq ConcreteFunction."""
        from tensorpotential.uq import constants as uq_constants

        uq_outputs = set(uq_sig.structured_outputs.keys())
        extra = [k for k in uq_constants.UQ_EXTRA_KEYS if k in uq_outputs]
        truncate = [k for k in uq_constants.UQ_TRUNCATE_KEYS if k in uq_outputs]
        self._set_uq_extra_properties(extra, truncate)

    def load_uq_artifacts(
        self,
        artifact_path: str,
        model_yaml: str = None,
        checkpoint: str = None,
        param_dtype=None,
        compute_dsigma_dr: bool = True,
    ):
        """Load GMM-UQ artifacts and enable uncertainty quantification.

        For saved models (the common case), ``model_yaml`` and ``checkpoint``
        are required to rebuild the model with the UQ compute function.
        For TPModel instances they are optional.

        ``compute_dsigma_dr`` selects between the full HAL-style compute
        (per-atom uncertainty forces and virials, default) and the cheaper
        gamma-only compute (scalar sigma/gamma per atom only).

        After this call, ``calculate()`` populates ``self.results`` with UQ
        keys (``atomic_sigma``, ``total_sigma``, ``dsigma_dr``, etc.) and
        ``self.gmm_uq_model`` is available for incremental updates / saving.
        """
        from tensorpotential.uq.gmmuq import GMMUQModel
        from tensorpotential.uq.compute import gmm_uq_compute_class
        from tensorpotential.uq.factories import (
            build_uq_compute_from_yaml,
            default_uq_extra_properties,
            patch_instructions_for_basis_rp_features,
            _basis_rp_spec_from_artifact,
        )

        if len(self.models) != 1:
            raise ValueError(
                "load_uq_artifacts requires exactly one model (ensembles not supported)"
            )
        model = self.models[0]

        # Path A: saved model (frozen graph) — rebuild from source
        if not hasattr(model, "compute_function"):
            if model_yaml is None or checkpoint is None:
                raise ValueError(
                    "model_yaml and checkpoint are required when the "
                    "calculator was created from a saved model"
                )
            tp_uq, gmm_uq, _, _ = build_uq_compute_from_yaml(
                model_yaml=model_yaml,
                checkpoint=checkpoint,
                gmm_artifact_path=artifact_path,
                param_dtype=param_dtype,
                compute_dsigma_dr=compute_dsigma_dr,
            )
            self.models[0] = tp_uq.model
            self.data_keys = list(tp_uq.model.compute_specs.keys())

        # Path B: TPModel (mutable) — swap compute_function and retrace
        else:
            if param_dtype is None:
                param_dtype = model.param_dtype
            gmm_uq = GMMUQModel(artifact_path, param_dtype=param_dtype)
            spec = _basis_rp_spec_from_artifact(gmm_uq)
            if spec is None or spec.get("matrix") is None:
                raise ValueError(
                    f"UQ artifact '{artifact_path}' was not built with the "
                    "basis-RP feature (missing uq_rp_matrix); rebuild with "
                    "`grace_uq build`."
                )
            compute_uq = gmm_uq_compute_class(compute_dsigma_dr)(gmm_uq_model=gmm_uq)
            rp = patch_instructions_for_basis_rp_features(
                model.instructions,
                out_dim=spec["out_dim"],
                seed=spec["seed"],
                projection_matrix=spec["matrix"],
                feature_transform=spec.get("transform"),
                normalize=spec.get("normalize", False),
                add_density_channel=spec.get("add_density_channel", False),
                density_scale=spec.get("density_scale", 1.0),
            )
            # Build the just-appended projection instruction (R as a tf.constant)
            # before retracing — the model is already built, so model.build()
            # would skip it and decorate_compute_function does not build.
            rp.build(param_dtype)
            model.compute_function = compute_uq
            # Drop instance attribute so decorate_compute_function re-wraps the
            # class method against the new compute_function (the old tf.function
            # has a stale trace).
            del model.compute
            model.decorate_compute_function(jit_compile=True)
            self.data_keys = list(model.compute_specs.keys())

        self.gmm_uq_model = gmm_uq
        extra, truncate = default_uq_extra_properties(
            gmm_uq, compute_dsigma_dr=compute_dsigma_dr,
        )
        self._set_uq_extra_properties(extra, truncate)

    def get_data(self, atoms):
        current_symbs = atoms.symbols.species()
        assert all([x in self.element_map for x in current_symbs]), (
            f"This model is configured to process "
            f"the following elements only: {list(self.element_map.keys())}, but the structure "
            f"contains {current_symbs}"
        )

        # Dense (reshape) aggregation: the DensePaddingManager selects a per-atom-uniform
        # (atoms, width) shape (reusing visited shapes to bound XLA recompiles for sequential
        # eval of heterogeneous structures) and emits the [n_atoms*width] layout via the
        # builder's dense pad_batch. Otherwise the segment_sum PaddingManager handles padding.
        if self.dense_reshape:
            data = self.dense_padding_manager.get_data(atoms)
        else:
            data = self.padding_manager.get_data(atoms)

        self.data = data

        # filtering
        data = {k: v for k, v in data.items() if k in self.data_keys}
        self.current_min_dist = np.min(
            np.linalg.norm(data[constants.BOND_VECTOR], axis=1)
        )
        if self.min_dist is not None and self.current_min_dist < self.min_dist:
            raise RuntimeError(
                f"Minimal bond distance {self.current_min_dist} is smaller than {self.min_dist}"
            )
        self.data = Dataset.from_tensors(data).get_single_element()

    def find_upper_padding_bound(self, max_nat):
        # find first entry, when max_nat<=
        for ind, upper_bound_tuple in enumerate(self.padding_history):
            if upper_bound_tuple[0] >= max_nat:
                break
        # shift further, until same max_nat
        while (
            ind + 1 < len(self.padding_history)
            and self.padding_history[ind + 1][0] == upper_bound_tuple[0]
        ):
            ind += 1
        upper_bound_tuple = self.padding_history[ind]
        return upper_bound_tuple

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces", "stress"),
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.forces = np.empty((len(atoms), 3))
        self.stress = np.empty((1, 3, 3))
        self.energy = 0.0
        results = {}

        self.get_data(atoms)

        t0 = time.perf_counter()

        self.outputs = []
        energy_list = []
        forces_list = []
        stress_list = []
        if self.extra_properties is not None:
            extras = {prop: [] for prop in self.extra_properties}
        else:
            extras = {}
        n_model = len(self.models)

        for model in self.models:
            output = model.compute(self.data)
            self.outputs.append(output)

        for output in self.outputs:
            if "energy" in self.compute_properties:
                e = output.get(constants.PREDICT_TOTAL_ENERGY, None)
                if e is None:
                    energy_list.append(np.zeros((1, 1)))
                else:
                    energy_list.append(e.numpy().astype(np.float64))

            if "forces" in self.compute_properties:
                forces = output.get(constants.PREDICT_FORCES, None)
                if forces is None:
                    forces = np.zeros((len(atoms), 3))
                else:
                    forces = forces.numpy()
                forces_list.append(forces)

            if "stress" in self.compute_properties:
                stress = output.get(constants.PREDICT_VIRIAL, None)
                if stress is None or atoms.get_cell().rank == 0:
                    stress = np.zeros((6,))
                else:
                    stress = (
                        -stress.numpy().reshape((6,))[[0, 1, 2, 5, 4, 3]]
                        / atoms.get_volume()
                    )
                stress_list.append(stress)
            for prop in extras:
                res = output.get(prop)
                if res is None:
                    extras[prop].append(np.zeros((1, 1)))
                else:
                    res = res.numpy()
                    t = self.truncate_extras_by_natoms
                    should_truncate = t is True or (
                        isinstance(t, list) and prop in t
                    )
                    if should_truncate:
                        extras[prop].append(res[: len(atoms)])
                    else:
                        extras[prop].append(res)
        self.eval_time = time.perf_counter() - t0

        # Feed the dense ski-rental promotion: eval_time wraps model.compute + the result-sync, so
        # the first eval of a freshly-minted shape carries its XLA compile and later evals are
        # steady -- exactly what calibrates compile-cost vs per-eval padding waste.
        if self.dense_reshape and self.dense_padding_manager is not None:
            self.dense_padding_manager.note_eval(self.eval_time)

        self.energy = np.mean(energy_list, axis=0).flatten()[0]

        # ensure only real atoms have forces
        self.forces = np.mean(forces_list, axis=0)[: len(atoms)]
        self.stress = np.mean(stress_list, axis=0)

        results["energy"] = self.energy
        results["free_energy"] = results["energy"]
        results["forces"] = self.forces
        results["stress"] = self.stress

        for k, v in extras.items():
            results[k] = np.mean(v, axis=0)

        if n_model > 1:
            results["energy_std"] = np.std(energy_list, axis=0).flatten()[0]
            results["forces_std"] = np.std(forces_list, axis=0)[: len(atoms)]
            results["stress_std"] = np.std(stress_list, axis=0)

        self.results = results
