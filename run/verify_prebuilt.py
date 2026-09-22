"""Verify prebuild_michelson gives identical XYZ to the original path, and time both.

Run on a GPU node for representative numbers:

    qsub -I -q interactive_gpu -l select=1:ncpus=4:ngpus=1 -l walltime=1:00:00
    module load cuda12.4/toolkit/12.4.1
    source /scratch/e1498138/anaconda3/bin/activate egb-multi
    python /home/svu/e1498138/localgit/eGB-multi/run/verify_prebuilt.py
"""

import json
import time
import h5py
import numpy as np

from egb_jax_eccentric import (
    EccentricBinaryParams,
    eccentric_xyz_jax,
    default_lisaorbits,
    state_from_lisaorbits,
    precompute_jax_link_geometry,
    prebuild_michelson,
)

inj_path = "/scratch/e1498138/eGB-multi/sangria_hmcnc_true.h5"
XYZ = ("X", "Y", "Z")
GENERATION = 1
MEASUREMENT_ORDER = 3
DELAY_ORDER = 3

with h5py.File(inj_path, "r") as f:
    t = f["t"][:]
    source_params = json.loads(f.attrs["egb_source_params_json"])

print("samples:", t.size)

state = state_from_lisaorbits(default_lisaorbits("equal"), t)
geometry = precompute_jax_link_geometry(state)


def make_source(ecc, m1, m2):
    return EccentricBinaryParams(
        mean_motion=np.pi * source_params["f0_hz"],
        eccentricity=ecc,
        m1_solar=m1,
        m2_solar=m2,
        distance_m=source_params["distance_m"],
        beta=source_params["beta"],
        lambda_=source_params["lambda_"],
        psi=source_params["psi"],
        inclination=source_params["inclination"],
        phi0=source_params["phi0"],
        fdot=source_params["fdot"],
    )


kwargs = dict(
    geometry=geometry,
    batch_size=1,
    physics_mode="1pn_periastron",
    generation=GENERATION,
    measurement_order=MEASUREMENT_ORDER,
    delay_order=DELAY_ORDER,
)

truth = make_source(
    source_params["eccentricity"], source_params["m1_solar"], source_params["m2_solar"]
)

print("\n--- warmup (JIT) ---")
eccentric_xyz_jax(state, truth, **kwargs)

print("\n--- prebuilding michelson combinations (one-off cost) ---")
t0 = time.time()
prebuilt = prebuild_michelson(state, generation=GENERATION, delay_order=DELAY_ORDER)
print("prebuild took:", round(time.time() - t0, 3), "s")

# --- correctness: original vs prebuilt, on several different sources ---
print("\n--- correctness check ---")
for ecc, m1, m2 in [(0.1, 0.55, 0.27), (0.35, 0.9, 0.4), (0.0, 0.3, 1.1)]:
    src = make_source(ecc, m1, m2)
    ref = eccentric_xyz_jax(state, src, **kwargs)
    new = eccentric_xyz_jax(state, src, prebuilt=prebuilt, **kwargs)
    worst = 0.0
    for ch in XYZ:
        a = np.asarray(ref[ch])
        b = np.asarray(new[ch])
        denom = max(float(np.max(np.abs(a))), np.finfo(float).tiny)
        worst = max(worst, float(np.max(np.abs(a - b))) / denom)
    status = "IDENTICAL" if worst == 0.0 else f"max rel diff {worst:.3e}"
    print(f"  e={ecc}, m1={m1}, m2={m2}: {status}")

# --- timing ---
print("\n--- timing (3 calls each, steady state) ---")


def timeit(fn, n=3):
    fn()  # warm
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n


old = timeit(lambda: eccentric_xyz_jax(state, truth, **kwargs))
new = timeit(lambda: eccentric_xyz_jax(state, truth, prebuilt=prebuilt, **kwargs))
print(f"  original : {old:.3f} s/call")
print(f"  prebuilt : {new:.3f} s/call")
print(f"  speedup  : {old / new:.2f}x")
