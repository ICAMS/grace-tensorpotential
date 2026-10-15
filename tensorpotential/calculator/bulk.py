"""Evaluate a GRACE (or any ASE) calculator over many structures.

Looping a calculator over a list of ``ase.Atoms`` is the single most repeated
snippet around this package — the CLIs do it, and so does every analysis script.
The order it runs in costs time, too: every ``(n_atoms, n_neighbours)`` padded
shape the model has not seen yet triggers an XLA compile. ``TPCalculator``'s
padding managers already bound how often that happens; feeding the LARGEST
structure first bounds it further, since the widest shape is then compiled once
and every smaller structure reuses it.

``predict_structures`` is that loop, with the ordering handled internally and the
results always returned in the caller's order. It runs one structure per forward
pass; for many small structures on a GPU the batched path in
``tensorpotential.uq.feature_extraction`` is faster, but that one is
``TPCalculator``-only whereas this works with any ASE calculator.
"""

from __future__ import annotations

import logging

import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError

log = logging.getLogger(__name__)

# get_* accessor per requested property
_ACCESSORS = {
    "energy": lambda at: float(at.get_potential_energy()),
    "forces": lambda at: np.asarray(at.get_forces()),
    "stress": lambda at: np.asarray(at.get_stress()),
}

# per-structure failures to log before falling silent; a total is logged at the end
_MAX_WARNINGS = 3


def predict_structures(
    atoms,
    calc,
    *,
    properties=("energy", "forces"),
    extra=(),
    optional=("stress",),
    sort_by_natoms=True,
    on_error="raise",
    progress=None,
    postprocess=None,
):
    """Evaluate ``calc`` on every structure in ``atoms``.

    Parameters
    ----------
    atoms : iterable of ase.Atoms
        Structures to evaluate. They are **copied**; the caller's objects never
        have a calculator attached and are not modified.
    calc : ase.calculators.calculator.Calculator
        Any ASE calculator — ``TPCalculator``, ``PyGRACEFSCalculator``, ...
        A single instance is reused for every structure.
    properties : sequence of str
        Any of ``energy``, ``forces``, ``stress``.
    extra : sequence of str
        Keys read out of ``calc.results`` after evaluation — e.g. ``gamma``,
        ``atomic_sigma``, ``features`` for a UQ-enabled model. These have to be
        harvested from ``results`` rather than requested as properties because
        ``TPCalculator`` does not declare its UQ outputs in
        ``implemented_properties`` (it builds a per-instance ``extra_properties``
        list instead); once it does, they can move into ``properties``.
        A key the calculator does not provide yields ``None`` for that structure
        rather than an error, since availability depends on the model.
    optional : sequence of str
        Properties this calculator may legitimately be unable to produce, giving
        ``None`` for that structure instead of an error. Defaults to
        ``("stress",)``: a calculator that does not implement stress is not a
        failure. Only ASE's ``PropertyNotImplementedError`` is tolerated — a
        stress computation that genuinely blows up still propagates.
    sort_by_natoms : bool
        Evaluate largest-first so the biggest XLA shape compiles once. This is a
        pure performance knob — results come back in the ORDER OF ``atoms``
        either way.
    on_error : {"raise", "warn"}
        What to do when a structure fails. ``warn`` logs the first few failures
        and stores ``None`` for that structure.
    progress : callable(done, total) or None
        Called after each structure. Use for a progress bar or log line.
    postprocess : callable(dict) -> dict, or None
        Applied per structure to the harvested ``extra`` values (key -> array, keys
        the calculator did not provide are absent) before they are stored; the
        returned dict replaces them and may drop keys or add new ones (new keys get
        a ``None`` entry for every other structure). Use it to keep only part of a
        large per-atom array (e.g. the features of some atoms) so the full array
        never outlives its structure.

    Returns
    -------
    dict of str -> list
        One key per requested property and per ``extra`` key. Every list has
        ``len(atoms)`` entries, aligned with the input; failed or unavailable
        entries are ``None``.

    Examples
    --------
    >>> out = predict_structures(structures, calc)                # doctest: +SKIP
    >>> np.array(out["energy"])                                   # doctest: +SKIP

    >>> out = predict_structures(structures, uq_calc,             # doctest: +SKIP
    ...                          properties=("energy",), extra=("gamma",))
    """
    if on_error not in ("raise", "warn"):
        raise ValueError(f"on_error must be 'raise' or 'warn', got {on_error!r}")
    unknown = set(properties) - set(_ACCESSORS)
    if unknown:
        raise ValueError(
            f"unknown properties {sorted(unknown)}; known: {sorted(_ACCESSORS)}"
        )

    atoms = list(atoms)
    total = len(atoms)
    out = {k: [None] * total for k in list(properties) + list(extra)}

    order = range(total)
    if sort_by_natoms:
        # largest first: the widest padded shape is compiled once and reused
        order = sorted(order, key=lambda i: -len(atoms[i]))

    optional = set(optional)
    n_failed = 0
    for done, i in enumerate(order, start=1):
        at = atoms[i].copy()  # never attach a calculator to the caller's object
        at.calc = calc
        try:
            for prop in properties:
                try:
                    out[prop][i] = _ACCESSORS[prop](at)
                except PropertyNotImplementedError:
                    # this calculator cannot produce the property at all, which
                    # for stress is a fact about the calculator rather than a
                    # failed structure. Deliberately NOT the builtin
                    # NotImplementedError: the package raises that from ~66
                    # sites, several on the forward pass, and swallowing those
                    # would hide real model errors as "stress unavailable".
                    if prop not in optional:
                        raise
            results = getattr(calc, "results", None) or {}
            values = {key: np.asarray(results[key]) for key in extra if key in results}
            if postprocess is not None:
                values = postprocess(values)
            for key, v in values.items():
                out.setdefault(key, [None] * total)[i] = v
        except Exception as exc:
            if on_error == "raise":
                raise
            n_failed += 1
            if n_failed <= _MAX_WARNINGS:
                log.warning("prediction failed for n_atoms=%d: %s", len(at), exc)
        if progress is not None:
            progress(done, total)

    if n_failed:
        log.warning("prediction failed for %d of %d structures", n_failed, total)
    return out
