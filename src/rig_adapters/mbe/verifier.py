"""Different-physics reduced-order verifier for the MBE adapter (D7, R10).

Why this module exists (implementation-plan D7 / §6.6 / §15.2 task iv, R10)
==========================================================================

D7 forbids the process simulator from being *both* the gray-box physics prior
*and* its own independent verifier. The MBE adapter's ``physics_plugin`` is the
fast Arrhenius/regime path (``rig_adapters.mbe.adapter.evaluate_physics``), and
the high-fidelity kMC ``ZoneEnsemble`` shares physics lineage with it (both are
thermally-activated incorporation models over the same geometry), so **neither
can verify the other**. Until now the adapter honestly returned
``independent_verifier=None``. This module supplies the owed *genuinely
different-physics* second model so a non-circular physics-fidelity check exists.

What it is: a purely geometric line-of-sight deposition ROM
===========================================================

:class:`GeometricDepositionVerifier` predicts the **flux-scale** channel
(``thickness_grown``) from a Knudsen effusion / view-factor calculation over the
*source and wafer geometry alone*. It contains **zero** thermal, kinetic, or
regime physics — no Arrhenius ``exp(-E/kT)``, no thermal solve, no kMC, no
material card, no fitted constant, and no line of code shared with the fast
path. It is an INDEPENDENT SANITY BAND, not a second simulator: it agrees with
an uncorrupted machine within a stated tolerance and disagrees loudly when the
machine's *delivered flux* is corrupted.

Geometry (source frame; wafer in the z=0 plane facing +z toward the source):

- effusion cell at ``S = (a, 0, H)`` — radial offset ``a = source_offset`` from
  the rotation axis, axial height ``H = source_height`` — aimed at the wafer
  point ``A = (b, 0, 0)``, ``b = aim_offset``. Cell-axis unit vector
  ``u_hat = (A - S)/La``, ``La = sqrt((b-a)^2 + H^2)``.
- a wafer point at radius ``r``, azimuth ``phi`` sits at ``P = (r cos phi,
  r sin phi, 0)``; the source->point ray is ``w = P - S`` with length
  ``D = sqrt(r^2 - 2 a r cos phi + a^2 + H^2)``.

Deposition flux delivered to that point (Knudsen cosine^m emission, inverse-
square spreading, cosine incidence on the flat wafer normal +z):

    cos_e = [ (b-a)(r cos phi - a) + H^2 ] / (La * D)      emission angle
    cos_i = H / D                                          incidence angle
    dPhi(r, phi) = (cos_e)^m * cos_i / D^2                 (0 if cos_e <= 0)

with ``m`` the ideal-Knudsen cosine exponent (``m = 1`` = Lambertian; the
verifier deliberately assumes the ideal cell lobe rather than reading the
machine's tunable ``cosine_n``, keeping it independent of that flux-shape knob).
Substrate rotation azimuthally averages the deposit, so the physically relevant
quantity is the ring-averaged ``Phi(r) = (1/2pi) integral dPhi dphi`` and its
area-weighted wafer mean ``Phi_bar``.

Delivery efficiency and the thickness prediction
------------------------------------------------

The machine models ``thickness_grown = film_thickness * flux_eff`` — the target
thickness times a *delivered-flux fraction* that a healthy, nominally-configured
source drives to 1 and that seasoning / source depletion / tool drift pull
below 1. The verifier reproduces the delivered-flux fraction geometrically as
``g = Phi_bar(machine_config) / Phi_bar(nominal_config)`` (``g = 1`` at the
nominal chamber build by construction) and predicts::

    thickness_pred = film_thickness * g

On the nominal build (the operating regime — the adapter holds machine config at
:data:`NOMINAL_MACHINE_CONFIG` as split-plot conditioning) this predicts the
target thickness, matching a clean machine to metrology precision; a hidden
flux-scale pathology drives the machine's ``thickness_grown`` away from the
target while the verifier — computing from clean geometry — still predicts it,
so the gap equals the flux loss and is flagged.

What it deliberately does NOT verify (honest scope)
---------------------------------------------------

- ``combined_nonuniformity_pct`` — on this machine the combined nonuniformity is
  ~98% Arrhenius *thermal* incorporation (measured: thermal ~78% vs geometric
  flux ~1.6% at the nominal recipe), i.e. exactly the physics this ROM omits by
  design. The verifier reports its own geometric ``flux_nonuniformity_pct`` as an
  informational diagnostic, but it is NOT a bound on the machine's (thermal-
  dominated) combined channel and must not be scored as one.
- ``slip_max_ratio``, ``T_center``, ``bow_cooldown_um`` — thermal-gradient /
  thermo-mechanical outputs with no flux-scale content; a geometric deposition
  model has nothing to say about them.
- **Off-nominal chamber rebuilds.** The delivery integral ``g(config)`` is
  physically correct (a real tool's delivered flux *does* fall as the source
  recedes, ~1/H^2), but this in-silico machine models ``thickness_grown`` as
  geometry-INDEPENDENT (``target * flux_eff``), so verifier and machine agree on
  thickness only at the calibrated nominal build. The verifier is therefore
  scoped to the nominal build; overriding ``machine_config`` models a *different*
  build (valid physics) that this machine's simplified thickness law will not
  track. Comparisons against the machine use the nominal config.

Determinism: pure closed-form geometry over a fixed quadrature — no RNG, no
state; identical inputs give bit-identical outputs.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

# --- Nominal chamber build (MIRRORS rig_adapters.mbe.adapter, asserted by a
# test). Kept as a local copy so this module imports NOTHING from the fast path:
# the "different-physics" claim is only credible if it shares no code with it. ---
NOMINAL_MACHINE_CONFIG: dict[str, float] = {
    "heater_radius": 0.018,
    "gap": 0.012,
    "source_offset": 0.10,
    "source_height": 0.20,
    "aim_offset": 0.0,
}

# Wafer radius [m]: the fast path builds ``UniformityProblem`` with its default
# R=0.0254 (1-inch), so the verifier bounds that same wafer. (Asserted by a test
# that reads the sim default via the adapter path.)
WAFER_RADIUS_M = 0.0254

# Ideal Knudsen (Lambertian) emission exponent — the ROM's own assumption, NOT
# the machine's tunable ``cosine_n``.
KNUDSEN_COSINE_M = 1.0

# Default agreement band on the thickness ratio. Physics/noise-derived, NOT tuned
# to an observed gap (verifier vs a clean machine agree exactly at nominal):
# the in-silico metrology noise on ``thickness_grown`` is sigma = a + b|y| with
# a=1e-9 m, b=2e-3, i.e. <=~0.7% rel at the thinnest films and ~0.3% at 1 um, so
# ~2% at 3 sigma; a meaningful flux pathology (seasoning/depletion/tool drift) is
# >=~10%. 5% sits cleanly between: a clean machine never trips it, a real flux
# loss always does.
DEFAULT_REL_BAND = 0.05

THICKNESS_CHANNEL = "thickness_grown"
FILM_THICKNESS_KEY = "film_thickness"

# Channels this ROM cannot bound (documented above); surfaced on every report so
# a consumer never mistakes silence for a pass.
UNBOUNDED_CHANNELS: tuple[str, ...] = (
    "combined_nonuniformity_pct",
    "slip_max_ratio",
    "T_center",
    "bow_cooldown_um",
)


@dataclass(frozen=True)
class VerifierReport:
    """Per-channel verdict of an independent-verifier check (D7).

    ``agrees`` is the headline: True iff every channel the ROM can *bound* is
    within band. ``bounded_channels`` are the flux-scale channels actually
    checked; ``unbounded_channels`` are surfaced explicitly so a caller cannot
    read a pass as blanket certification (§3.4 the gate is only as strong as the
    verifier's real reach).
    """

    agrees: bool
    predicted_thickness_m: float
    observed_thickness_m: float
    thickness_rel_error: float
    band: float
    within_band: bool
    delivery_ratio: float
    bounded_channels: tuple[str, ...] = (THICKNESS_CHANNEL,)
    unbounded_channels: tuple[str, ...] = UNBOUNDED_CHANNELS
    detail: str = ""


def outcome_values(observed: Any) -> dict[str, float]:
    """Normalize an observation to ``{output_name: SI magnitude}``.

    Accepts a plain ``Mapping[str, float]``, a ``RunRecord`` (anything with an
    ``.outcomes`` attribute), or an iterable of ``OutcomeRecord``-shaped objects
    (``.name`` + ``.value.magnitude``). Duck-typed so this module imports nothing
    from ``rig.schema`` or the adapter.
    """
    if isinstance(observed, Mapping):
        return {str(k): float(v) for k, v in observed.items()}
    outcomes = getattr(observed, "outcomes", observed)
    values: dict[str, float] = {}
    for o in outcomes:  # OutcomeRecord-shaped
        name = getattr(o, "name")  # noqa: B009 - explicit, fail loud if absent
        mag = getattr(getattr(o, "value"), "magnitude")  # noqa: B009
        values[str(name)] = float(mag)
    return values


class GeometricDepositionVerifier:
    """Purely-geometric line-of-sight deposition ROM (D7 independent verifier).

    Callable as ``verifier(recipe, machine_config=None) -> dict`` (the
    :class:`~rig.interfaces.ProcessAdapter` ``independent_verifier`` slot expects
    a ``Callable``), plus :meth:`predict` / :meth:`check` for structured use.
    """

    def __init__(
        self,
        *,
        nominal_config: Mapping[str, float] | None = None,
        wafer_radius_m: float = WAFER_RADIUS_M,
        cosine_m: float = KNUDSEN_COSINE_M,
        rel_band: float = DEFAULT_REL_BAND,
        n_r: int = 48,
        n_phi: int = 64,
    ) -> None:
        if rel_band <= 0.0:
            raise ValueError("rel_band must be > 0")
        if n_r < 3 or n_phi < 4:
            raise ValueError("quadrature too coarse (need n_r>=3, n_phi>=4)")
        self.nominal_config = dict(NOMINAL_MACHINE_CONFIG)
        self.nominal_config.update(nominal_config or {})
        self.wafer_radius_m = float(wafer_radius_m)
        self.cosine_m = float(cosine_m)
        self.rel_band = float(rel_band)
        self._n_r = int(n_r)
        self._n_phi = int(n_phi)
        # Precompute the nominal areal mean flux once (the denominator of every
        # delivery ratio). Purely geometric; no state beyond this cache.
        self._phi_bar_nominal = self._areal_mean_flux(self.nominal_config)

    # -- geometry ------------------------------------------------------------
    def _ring_flux(self, cfg: Mapping[str, float], r: float) -> float:
        """Rotation-averaged relative deposition flux at wafer radius ``r``."""
        a = float(cfg["source_offset"])
        h = float(cfg["source_height"])
        b = float(cfg.get("aim_offset", 0.0))
        m = self.cosine_m
        la = math.hypot(a - b, h)
        acc = 0.0
        for k in range(self._n_phi):
            phi = 2.0 * math.pi * (k + 0.5) / self._n_phi
            cphi = math.cos(phi)
            d2 = r * r - 2.0 * a * r * cphi + a * a + h * h
            d = math.sqrt(d2)
            cos_e = ((b - a) * (r * cphi - a) + h * h) / (la * d)
            if cos_e <= 0.0:
                continue  # wafer point is behind the cell's emission cone
            cos_i = h / d
            acc += (cos_e**m) * cos_i / d2
        return acc / self._n_phi

    def _flux_profile(self, cfg: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
        r_nodes = np.linspace(0.0, self.wafer_radius_m, self._n_r)
        prof = np.array([self._ring_flux(cfg, float(r)) for r in r_nodes])
        return r_nodes, prof

    def _areal_mean_flux(self, cfg: Mapping[str, float]) -> float:
        """Area-weighted mean of Phi(r) over the wafer disk."""
        r, prof = self._flux_profile(cfg)
        shells = r[1:] ** 2 - r[:-1] ** 2
        return float(np.sum(0.5 * (prof[1:] + prof[:-1]) * shells) / np.sum(shells))

    @staticmethod
    def _nonuniformity_pct(r: np.ndarray, prof: np.ndarray) -> float:
        shells = r[1:] ** 2 - r[:-1] ** 2
        mean = float(np.sum(0.5 * (prof[1:] + prof[:-1]) * shells) / np.sum(shells))
        if mean == 0.0:
            return 0.0
        return 100.0 * float(prof.max() - prof.min()) / mean

    def delivery_ratio(self, machine_config: Mapping[str, float] | None = None) -> float:
        """Geometric delivered-flux fraction relative to the nominal build.

        ``1.0`` at the nominal chamber build; falls as the source recedes /
        moves off-axis (the ~1/H^2 line-of-sight law). This is the quantity the
        machine calls ``flux_eff`` for a *healthy* tool.
        """
        if machine_config is None:
            return 1.0
        cfg = dict(self.nominal_config)
        cfg.update(machine_config)
        return self._areal_mean_flux(cfg) / self._phi_bar_nominal

    # -- prediction ----------------------------------------------------------
    def predict(
        self,
        recipe: Mapping[str, Any],
        machine_config: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Predict the flux-scale channel from geometry + recipe.

        Returns ``thickness_grown`` (the validated channel),
        ``delivery_ratio`` (the geometric flux fraction), and
        ``flux_nonuniformity_pct`` (an informational diagnostic — see the module
        docstring; NOT a bound on the machine's combined nonuniformity).
        """
        if FILM_THICKNESS_KEY not in recipe:
            raise ValueError(f"recipe is missing required variable {FILM_THICKNESS_KEY!r}")
        film_thickness = float(
            getattr(recipe[FILM_THICKNESS_KEY], "magnitude", recipe[FILM_THICKNESS_KEY])
        )
        cfg = dict(self.nominal_config)
        cfg.update(machine_config or {})
        g = self._areal_mean_flux(cfg) / self._phi_bar_nominal
        r, prof = self._flux_profile(cfg)
        return {
            THICKNESS_CHANNEL: film_thickness * g,
            "delivery_ratio": g,
            "flux_nonuniformity_pct": self._nonuniformity_pct(r, prof),
        }

    def __call__(
        self,
        recipe: Mapping[str, Any],
        machine_config: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        # ProcessAdapter.independent_verifier is a bare Callable; predict IS it.
        return self.predict(recipe, machine_config)

    # -- verification --------------------------------------------------------
    def check(
        self,
        recipe: Mapping[str, Any],
        observed: Any,
        machine_config: Mapping[str, float] | None = None,
    ) -> VerifierReport:
        """Independently check a machine observation of ``recipe`` (D7).

        ``observed`` is a ``RunRecord`` / outcome list / ``{name: SI value}``
        mapping. Compares the machine's ``thickness_grown`` against the geometric
        prediction; ``agrees`` is True iff it is within ``rel_band``.
        """
        pred = self.predict(recipe, machine_config)
        predicted = pred[THICKNESS_CHANNEL]
        values = outcome_values(observed)
        if THICKNESS_CHANNEL not in values:
            raise ValueError(
                f"observation is missing the verifiable channel {THICKNESS_CHANNEL!r}; "
                f"got {sorted(values)}"
            )
        obs = values[THICKNESS_CHANNEL]
        if predicted == 0.0:
            rel_err = 0.0 if obs == 0.0 else math.inf
        else:
            rel_err = (obs - predicted) / predicted
        within = abs(rel_err) <= self.rel_band
        detail = (
            f"thickness_grown observed={obs:.4e} m vs geometric prediction "
            f"{predicted:.4e} m (delivery_ratio={pred['delivery_ratio']:.4f}); "
            f"rel_error={rel_err:+.3%} vs band +-{self.rel_band:.1%} -> "
            f"{'AGREE' if within else 'DISAGREE'}. Unbounded (thermal/mechanical, "
            f"out of scope): {', '.join(UNBOUNDED_CHANNELS)}."
        )
        return VerifierReport(
            agrees=within,
            predicted_thickness_m=predicted,
            observed_thickness_m=obs,
            thickness_rel_error=rel_err,
            band=self.rel_band,
            within_band=within,
            delivery_ratio=pred["delivery_ratio"],
            detail=detail,
        )


def verifier_imports(module_source: str) -> set[str]:
    """Top-level module names imported by ``module_source`` (AST, for tests).

    Lets a test mechanically assert the ROM pulls in nothing from the fast path
    (``mbe_sim``, ``simlink``, ``sources``, ``thermal``, ``wafer``, the adapter),
    i.e. that the different-physics claim is real and not just documentation.
    """
    tree = ast.parse(module_source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names
