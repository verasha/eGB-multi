"""pyTDI bridge for JAX eccentric link responses."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .eccentric import EccentricBinaryParams
from .eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry
from .geometry import LISAState

LINKS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 2),
    (2, 1),
    (1, 0),
)
# Rosetta Stone / pyTDI convention: label ij is received at i, emitted by j.
LINK_LABELS: tuple[str, ...] = ("21", "32", "13", "31", "23", "12")


def _sampling_frequency(t: NDArray[np.float64]) -> float:
    if t.size < 2:
        raise ValueError("at least two time samples are required")
    dt = np.diff(t)
    if not np.allclose(dt, dt[0], rtol=1e-9, atol=1e-12):
        raise ValueError("pyTDI data requires a uniformly sampled time grid")
    return float(1.0 / dt[0])


def pytdi_data_from_links(
    state: LISAState,
    links: dict[str, NDArray[np.complex128]],
    *,
    compute_delay_derivatives: bool = True,
):
    """Create a `pytdi.Data` object from six GW-only link responses.

    The link responses are assigned to `sci_ij`. Local metrology/reference
    measurements are set to zero, so pyTDI's intermediate variables reduce to
    the supplied GW link content.
    """

    try:
        from pytdi import Data
    except ImportError as exc:
        raise ImportError("pytdi is required for pytdi_data_from_links") from exc

    measurements = _measurements_from_links(links, state.t.size)

    delays = _delays_from_state(state)

    data = Data(measurements, delays, _sampling_frequency(state.t))
    if compute_delay_derivatives:
        data.compute_delay_derivatives()
    return data


def _measurements_from_links(links: dict[str, NDArray[np.complex128]], n_samples: int) -> dict[str, NDArray]:
    """Map six GW link responses onto pyTDI's `sci_ij`, zeroing everything else."""

    from pytdi import Data

    missing = sorted(set(LINK_LABELS) - set(links))
    if missing:
        raise ValueError(f"missing link responses for {missing}")

    zero = np.zeros(n_samples, dtype=np.complex128)
    measurements = {key: zero for key in Data.MEASUREMENTS}
    for label in LINK_LABELS:
        value = np.asarray(links[label], dtype=np.complex128)
        if value.shape != (n_samples,):
            raise ValueError(f"link {label} has shape {value.shape}, expected {(n_samples,)}")
        measurements[f"sci_{label}"] = value
    return measurements


def _delays_from_state(state: LISAState) -> dict[str, NDArray[np.float64]]:
    """Light-travel times per link, derived purely from the constellation geometry."""

    return {
        f"d_{label}": state.arm_length(sender, receiver)
        for label, (sender, receiver) in zip(LINK_LABELS, LINKS, strict=True)
    }


def _michelson_rot_links(rot: int) -> list[str]:
    if rot == 0:
        return ["12", "21", "13", "31"]
    if rot == 1:
        return ["23", "32", "21", "12"]
    if rot == 2:
        return ["31", "13", "32", "23"]
    raise ValueError(f"rot must be 0, 1 or 2 (got {rot})")


def prebuild_michelson(
    state: LISAState,
    *,
    generation: int = 1,
    delay_order: int = 3,
):
    """Prebuild the geometry-dependent parts of the Michelson X/Y/Z combinations.

    `pytdi.michelson.compute_factorized_michelson` rebuilds its combinations on
    every call, but `build()` depends only on the delays (i.e. on `state`), never
    on the source. For repeated evaluations over a fixed constellation -- e.g. a
    sampler varying source parameters -- the build can be hoisted out and reused,
    which is what this returns. Pair it with `xyz_from_prebuilt`.
    """

    if generation not in (1, 2):
        raise ValueError(f"invalid generation '{generation}', must be 1 or 2")

    try:
        from pytdi import Data, core, intervar
    except ImportError as exc:
        raise ImportError("pytdi is required for prebuild_michelson") from exc

    n_samples = state.t.size
    zero = np.zeros(n_samples, dtype=np.complex128)
    data = Data(
        {key: zero for key in Data.MEASUREMENTS},
        _delays_from_state(state),
        _sampling_frequency(state.t),
    )
    data.compute_delay_derivatives()
    args = data.args

    built: dict[str, tuple] = {}
    for rot, channel in enumerate(("X", "Y", "Z")):
        links = _michelson_rot_links(rot)
        arm_1 = (
            core.LISATDICombination(
                {f"eta_{links[0]}": [(1, [])], f"eta_{links[1]}": [(1, [f"D_{links[0]}"])]}
            )
            @ intervar.ETA_SET
        ).build(**args, order=delay_order)
        arm_2 = (
            core.LISATDICombination(
                {f"eta_{links[2]}": [(1, [])], f"eta_{links[3]}": [(1, [f"D_{links[2]}"])]}
            )
            @ intervar.ETA_SET
        ).build(**args, order=delay_order)

        if generation == 1:
            final = core.LISATDICombination(
                {
                    "x_arm_1": [(-1, []), (1, [f"D_{links[2]}", f"D_{links[3]}"])],
                    "x_arm_2": [(1, []), (-1, [f"D_{links[0]}", f"D_{links[1]}"])],
                }
            ).build(**args, order=delay_order)
            built[channel] = (arm_1, arm_2, final)
        else:
            roundtrip_1 = core.LISATDICombination(
                {"arm_a": [(1, [])], "arm_b": [(1, [f"D_{links[0]}", f"D_{links[1]}"])]}
            ).build(**args, order=delay_order)
            roundtrip_2 = core.LISATDICombination(
                {"arm_a": [(1, [f"D_{links[2]}", f"D_{links[3]}"])], "arm_b": [(1, [])]}
            ).build(**args, order=delay_order)
            final = core.LISATDICombination(
                {
                    "roundtrip_a": [
                        (-1, []),
                        (1, [f"D_{links[2]}", f"D_{links[3]}", f"D_{links[0]}", f"D_{links[1]}"]),
                    ],
                    "roundtrip_b": [
                        (1, []),
                        (-1, [f"D_{links[0]}", f"D_{links[1]}", f"D_{links[2]}", f"D_{links[3]}"]),
                    ],
                }
            ).build(**args, order=delay_order)
            built[channel] = (arm_1, arm_2, roundtrip_1, roundtrip_2, final)

    return {"generation": generation, "n_samples": n_samples, "built": built}


def xyz_from_prebuilt(
    prebuilt,
    links: dict[str, NDArray[np.complex128]],
    *,
    measurement_order: int = 3,
) -> dict[str, NDArray[np.complex128]]:
    """Evaluate prebuilt Michelson combinations (see `prebuild_michelson`) on new links."""

    generation = prebuilt["generation"]
    measurements = _measurements_from_links(links, prebuilt["n_samples"])

    xyz = {}
    for channel, combinations in prebuilt["built"].items():
        if generation == 1:
            arm_1, arm_2, final = combinations
        else:
            arm_1, arm_2, roundtrip_1, roundtrip_2, final = combinations

        x_arm_1 = arm_1(measurements, order=measurement_order, unit="frequency")
        x_arm_2 = arm_2(measurements, order=measurement_order, unit="frequency")

        if generation == 1:
            xyz[channel] = final(
                {"x_arm_1": x_arm_1, "x_arm_2": x_arm_2},
                order=measurement_order,
                unit="frequency",
            )
        else:
            arms = {"arm_a": x_arm_1, "arm_b": x_arm_2}
            xyz[channel] = final(
                {
                    "roundtrip_a": roundtrip_1(arms, order=measurement_order, unit="frequency"),
                    "roundtrip_b": roundtrip_2(arms, order=measurement_order, unit="frequency"),
                },
                order=measurement_order,
                unit="frequency",
            )
    return xyz


def xyz_from_links(
    state: LISAState,
    links: dict[str, NDArray[np.complex128]],
    *,
    generation: int = 1,
    measurement_order: int = 3,
    delay_order: int = 3,
) -> dict[str, NDArray[np.complex128]]:
    """Compute pyTDI Michelson X/Y/Z from six link responses."""

    try:
        from pytdi.michelson import compute_factorized_michelson
    except ImportError as exc:
        raise ImportError("pytdi is required for xyz_from_links") from exc

    data = pytdi_data_from_links(state, links)
    return {
        "X": compute_factorized_michelson(
            data,
            rot=0,
            order=measurement_order,
            delay_order=delay_order,
            generation=generation,
            unit="frequency",
        ),
        "Y": compute_factorized_michelson(
            data,
            rot=1,
            order=measurement_order,
            delay_order=delay_order,
            generation=generation,
            unit="frequency",
        ),
        "Z": compute_factorized_michelson(
            data,
            rot=2,
            order=measurement_order,
            delay_order=delay_order,
            generation=generation,
            unit="frequency",
        ),
    }


def aet_from_xyz(xyz: dict[str, NDArray]) -> dict[str, NDArray]:
    """Convert Michelson X/Y/Z to orthogonal A/E/T channels."""

    x = np.asarray(xyz["X"])
    y = np.asarray(xyz["Y"])
    z = np.asarray(xyz["Z"])
    return {
        "A": (z - x) / np.sqrt(2.0),
        "E": (x - 2.0 * y + z) / np.sqrt(6.0),
        "T": (x + y + z) / np.sqrt(3.0),
    }


def eccentric_xyz_jax(
    state: LISAState,
    sources: EccentricBinaryParams | list[EccentricBinaryParams],
    *,
    geometry: dict[str, NDArray[np.float64]] | None = None,
    prebuilt=None,
    batch_size: int = 1024,
    physics_mode: str = "1pn",
    generation: int = 1,
    measurement_order: int = 3,
    delay_order: int = 3,
) -> dict[str, NDArray[np.complex128]]:
    """Compute JAX eccentric links and route them through pyTDI Michelson XYZ.

    Pass `prebuilt=prebuild_michelson(state, ...)` to reuse the geometry-dependent
    pyTDI build across calls, in the same way `geometry=` reuses the JAX link
    geometry. Both are worth doing when evaluating many sources on one grid.
    """

    if geometry is None:
        geometry = precompute_jax_link_geometry(state)
    links = eccentric_links_jax(
        sources,
        geometry,
        batch_size=batch_size,
        physics_mode=physics_mode,
    )
    if prebuilt is not None:
        if prebuilt["generation"] != generation:
            raise ValueError(
                f"prebuilt is for generation {prebuilt['generation']}, but generation={generation} was requested"
            )
        return xyz_from_prebuilt(prebuilt, links, measurement_order=measurement_order)
    return xyz_from_links(
        state,
        links,
        generation=generation,
        measurement_order=measurement_order,
        delay_order=delay_order,
    )


def eccentric_aet_jax(
    state: LISAState,
    sources: EccentricBinaryParams | list[EccentricBinaryParams],
    **kwargs,
) -> dict[str, NDArray]:
    """Compute JAX eccentric links, pyTDI XYZ, and orthogonal A/E/T channels."""

    return aet_from_xyz(eccentric_xyz_jax(state, sources, **kwargs))
