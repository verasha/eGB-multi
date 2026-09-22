"""
Profile a single log_likelihood/htemp evaluation on a real GPU node.

Run this from an interactive GPU session (see instructions below), inside
the egb-multi conda env, so the timing/profile reflects actual GPU behavior
instead of the CPU-only approximation from the login node.

How to get an interactive GPU session and run this:

    qsub -I -q interactive_gpu -l select=1:ncpus=4:ngpus=1 -l walltime=1:00:00

Once you're dropped into a shell on the compute node:

    module load cuda12.4/toolkit/12.4.1
    source /scratch/e1498138/anaconda3/bin/activate egb-multi
    python /home/svu/e1498138/localgit/eGB-multi/run/profile_likelihood_gpu.py
"""

import cProfile
import pstats
import io
import json
import time
import h5py
import numpy as np

from egb_jax_eccentric import (
    EccentricBinaryParams,
    aet_from_xyz,
    eccentric_xyz_jax,
    default_lisaorbits,
    state_from_lisaorbits,
    precompute_jax_link_geometry,
)

import jax
print("jax devices:", jax.devices())
print("jax backend:", jax.default_backend())

inj_path = "/scratch/e1498138/eGB-multi/sangria_hmcnc_true.h5"
XYZ = ("X", "Y", "Z")

with h5py.File(inj_path, "r") as f:
    t = f["t"][:]
    dt = float(f.attrs["segment_dt_s"])
    source_params = json.loads(f.attrs["egb_source_params_json"])

print("samples:", t.size, "dt:", dt)

state = state_from_lisaorbits(default_lisaorbits("equal"), t)
geometry = precompute_jax_link_geometry(state)

egb_pytdi_gen = 1
measurement_order = 3
delay_order = 3


def htemp(params):
    ecc_i, m1_sol_i, m2_sol_i = params
    source_temp = EccentricBinaryParams(
        mean_motion=np.pi * source_params["f0_hz"],
        eccentricity=ecc_i,
        m1_solar=m1_sol_i,
        m2_solar=m2_sol_i,
        distance_m=source_params["distance_m"],
        beta=source_params["beta"],
        lambda_=source_params["lambda_"],
        psi=source_params["psi"],
        inclination=source_params["inclination"],
        phi0=source_params["phi0"],
        fdot=source_params["fdot"],
    )
    raw_temp_xyz = eccentric_xyz_jax(
        state, source_temp, geometry=geometry, batch_size=1,
        physics_mode="1pn_periastron", generation=egb_pytdi_gen,
        measurement_order=measurement_order, delay_order=delay_order,
    )
    xyz = {ch: np.real(np.asarray(raw_temp_xyz[ch], dtype=np.complex128)) for ch in XYZ}
    return aet_from_xyz(xyz)


theta_true = (source_params["eccentricity"], source_params["m1_solar"], source_params["m2_solar"])

print("\n--- warmup call (pays JIT compile cost) ---")
t0 = time.time()
htemp(theta_true)
print("warmup call took:", time.time() - t0, "seconds")

print("\n--- timed steady-state call ---")
t0 = time.time()
htemp(theta_true)
print("steady-state call took:", time.time() - t0, "seconds")

print("\n--- profiled call (cProfile breakdown) ---")
profiler = cProfile.Profile()
profiler.enable()
htemp(theta_true)
profiler.disable()

stream = io.StringIO()
stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
stats.print_stats(30)
print(stream.getvalue())
