"""Build an HM Cnc injection into Sangria noise at a chosen eccentricity.

    ECC=0.3 SCALE=10 python run/make_injection.py

Writes the same layout the recovery scripts expect: 'noise', 'eccentric_signal'
(UNSCALED) and 'injected' (= noise + SCALE*signal), with the scale recorded in
the attrs so the template can be matched to it.
"""

import json
import os

import astropy.units as u
import h5py
import numpy as np
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord

from egb_jax_eccentric import (
    EccentricBinaryParams,
    default_lisaorbits,
    eccentric_links_jax,
    precompute_jax_link_geometry,
    prepare_xyz_from_links,
    state_from_lisaorbits,
)
from egb_jax_eccentric.constants import PARSEC_M

ECC = float(os.environ.get("ECC", 0.3))
SCALE = float(os.environ.get("SCALE", 10.0))
DURATION_DAYS = float(os.environ.get("DURATION_DAYS", 30.0))

scratch = "/scratch/e1498138/eGB-multi"
sangria = f"{scratch}/LDC2_sangria_training_v2.h5"
out_path = os.environ.get(
    "OUT", f"{scratch}/sangria_hmcnc_{int(SCALE)}x_e{str(ECC).replace('.', '_')}.h5")

XYZ = ("X", "Y", "Z")
SKY = ["sky/dgb/tdi", "sky/igb/tdi", "sky/vgb/tdi", "sky/mbhb/tdi"]
gen, measurement_order, delay_order = 1, 3, 3

# HM Cnc
period_s, ra_deg, dec_deg = 321.529129, 121.5957, 15.4586
m1_sol, m2_sol, dist_pc, inc_deg = 0.55, 0.27, 7500.0, 38.0

print(f"ECC={ECC}  SCALE={SCALE}  DURATION_DAYS={DURATION_DAYS}")
print("out:", out_path, flush=True)


def read_group(h5, path, start, stop):
    block = h5[path][start:stop, 0]
    return {n: np.asarray(block[n], dtype=np.float64) for n in ("t", *XYZ)}


with h5py.File(sangria, "r") as h5:
    dset = h5["obs/tdi"]
    dt, t0 = float(dset.attrs["dt"]), float(dset.attrs["t0"])
    stop = min(int(round(DURATION_DAYS * 86400.0 / dt)), dset.shape[0])
    obs = read_group(h5, "obs/tdi", 0, stop)
    noise_xyz = {"t": obs["t"], **{c: obs[c].copy() for c in XYZ}}
    for path in SKY:
        comp = read_group(h5, path, 0, stop)
        for c in XYZ:
            noise_xyz[c] -= comp[c]
    sangria_metadata = {
        "obs_tdi_dt_s": dt, "obs_tdi_t0_s": t0,
        "obs_tdi_units": str(dset.attrs.get("units", "")),
        "tdi_generation": float(h5["instru/config/TDI_GENERATION"][()]),
    }

t = noise_xyz["t"]
print(f"samples: {t.size}  dt: {dt}", flush=True)

ecl = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").transform_to(
    BarycentricTrueEcliptic())
source_params = {
    "f0_hz": 2.0 / period_s,
    "eccentricity": ECC,
    "m1_solar": m1_sol,
    "m2_solar": m2_sol,
    "distance_m": dist_pc * PARSEC_M,
    "beta": float(ecl.lat.rad),
    "lambda_": float(ecl.lon.rad),
    "psi": 0.0,
    "inclination": np.radians(inc_deg),
    "phi0": 0.0,
    "fdot": 0.0,
}
print("source:", json.dumps(source_params, indent=2), flush=True)

state = state_from_lisaorbits(default_lisaorbits("equal"), t)
geometry = precompute_jax_link_geometry(state)
xyz_eval = prepare_xyz_from_links(
    state, generation=gen, measurement_order=measurement_order, delay_order=delay_order)

src = EccentricBinaryParams(
    mean_motion=np.pi * source_params["f0_hz"],
    eccentricity=source_params["eccentricity"],
    m1_solar=source_params["m1_solar"], m2_solar=source_params["m2_solar"],
    distance_m=source_params["distance_m"],
    beta=source_params["beta"], lambda_=source_params["lambda_"],
    psi=source_params["psi"], inclination=source_params["inclination"],
    phi0=source_params["phi0"], fdot=source_params["fdot"])

raw = xyz_eval(eccentric_links_jax(src, geometry, batch_size=1,
                                   physics_mode="1pn_periastron"))
signal_xyz = {c: np.real(np.asarray(raw[c], dtype=np.complex128)).astype(np.float64)
              for c in XYZ}
injected_xyz = {c: noise_xyz[c] + SCALE * signal_xyz[c] for c in XYZ}

print("signal max abs:", {c: float(np.max(np.abs(signal_xyz[c]))) for c in XYZ}, flush=True)

with h5py.File(out_path, "w") as out:
    out.attrs["segment_dt_s"] = float(dt)
    out.attrs["segment_start_s"] = float(t[0])
    out.attrs["segment_stop_s"] = float(t[-1])
    out.attrs["injection_scale"] = float(SCALE)
    out.attrs["egb_pytdi_generation"] = gen
    out.attrs["egb_source_params_json"] = json.dumps(source_params, default=float)
    out.attrs["sangria_metadata_json"] = json.dumps(sangria_metadata, default=float)
    out.attrs["input_hdf5"] = sangria
    out.create_dataset("t", data=t, compression="gzip")
    for name, series in (("noise", noise_xyz), ("eccentric_signal", signal_xyz),
                         ("injected", injected_xyz)):
        g = out.create_group(name)
        for c in XYZ:
            g.create_dataset(c, data=series[c], compression="gzip")

with h5py.File(out_path, "r") as chk:
    for c in XYZ:
        r = (np.max(np.abs(chk["injected"][c][:] - chk["noise"][c][:]))
             / np.max(np.abs(chk["eccentric_signal"][c][:])))
        assert abs(r - SCALE) < 1e-6, f"{c}: ratio {r} != SCALE {SCALE}"
print("\nwrote", out_path, "(scale ratio verified)")
