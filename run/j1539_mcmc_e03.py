"""Bare-bones random-walk MCMC for the ZTF J1539+5027 e=0.3 injection.

Same structure as run/j1539_mcmc_e00.py. The mass box is NOT the HM Cnc box:
Fisher gives sigma(m1)=2.03e-2 here, so the tight [0.56,0.66] box would sit only
2.5 sigma from the walls and clip the posterior -- the exact failure diagnosed in
the HM Cnc e=0.1 run. m1 is widened to 6 sigma; m2 was already interior at 5.1.
Proposal is scaled to sigma(Mc)=6.7e-4, about half HM Cnc's, hence 0.002."""

import json
import os
import sys
import numpy as np
import h5py
import jax
import jax.numpy as jnp
import blackjax
from scipy.signal import welch

from egb_jax_eccentric import (
    EccentricBinaryParams, aet_from_xyz, default_lisaorbits, eccentric_links_jax,
    precompute_jax_link_geometry, prepare_xyz_from_links, state_from_lisaorbits,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j1539_config import MASS_BOUNDS, MASS_SIGMA

INJ = "/scratch/e1498138/eGB-multi/sangria_j1539_10x_e03.h5"
OUT = "/scratch/e1498138/eGB-multi/j1539_mcmc_e03.npz"
N_STEPS = 15000
SIGMA = jnp.array([5e-4, MASS_SIGMA, MASS_SIGMA])
BOUNDS = np.array([[0.0, 0.60], *MASS_BOUNDS])
XYZ, CH = ("X", "Y", "Z"), ("A", "E")

# --- data
with h5py.File(INJ, "r") as f:
    t = f["t"][:]
    noise = aet_from_xyz({c: f["noise"][c][:] for c in XYZ})
    data = aet_from_xyz({c: f["injected"][c][:] for c in XYZ})
    dt = float(f.attrs["segment_dt_s"])
    sp = json.loads(f.attrs["egb_source_params_json"])
    scale = float(f.attrs["injection_scale"])

# --- likelihood pieces (all fixed, computed once)
state = state_from_lisaorbits(default_lisaorbits("equal"), t)
geom = precompute_jax_link_geometry(state)
xyz_eval = prepare_xyz_from_links(state, generation=1, measurement_order=3, delay_order=3)

freq, psd = {}, {}
for c in ("A", "E", "T"):
    freq, p = welch(noise[c], fs=1 / dt, nperseg=8192, detrend="constant")
    psd[c] = p

fgrid = np.fft.rfftfreq(t.size, d=dt)
i0 = int(np.searchsorted(fgrid, freq[freq > 0].min()))
df = fgrid[1] - fgrid[0]
d_w, w_hh = {}, {}
for c in CH:
    sn = np.exp(np.interp(np.log(fgrid[i0:]), np.log(freq[1:]), np.log(psd[c][1:])))
    d_w[c] = 4 * df * dt * (np.fft.rfft(data[c]) * dt)[i0:] / sn
    w_hh[c] = 4 * df * dt**2 / sn


def loglike(p):
    src = EccentricBinaryParams(
        mean_motion=np.pi * sp["f0_hz"], eccentricity=p[0], m1_solar=p[1], m2_solar=p[2],
        distance_m=sp["distance_m"], beta=sp["beta"], lambda_=sp["lambda_"],
        psi=sp["psi"], inclination=sp["inclination"], phi0=sp["phi0"], fdot=sp["fdot"])
    xyz = xyz_eval(eccentric_links_jax(src, geom, batch_size=1, physics_mode="1pn_periastron"))
    aet = aet_from_xyz({c: scale * np.real(np.asarray(xyz[c], dtype=np.complex128)) for c in XYZ})
    ll = 0.0
    for c in CH:
        h = np.fft.rfft(aet[c])[i0:]
        ll += np.vdot(h, d_w[c]).real - 0.5 * np.sum((h.real**2 + h.imag**2) * w_hh[c])
    return float(ll)


def logdensity(theta):
    p = np.asarray(theta, dtype=np.float64)
    if np.any(p < BOUNDS[:, 0]) or np.any(p > BOUNDS[:, 1]):
        return jnp.asarray(-np.inf)
    return jnp.asarray(loglike(p))


# --- sample
truth = np.array([sp["eccentricity"], sp["m1_solar"], sp["m2_solar"]])
print("logL at truth:", loglike(truth), flush=True)

kernel = blackjax.additive_step_random_walk.normal_random_walk(logdensity, sigma=SIGMA)
st = kernel.init(jnp.asarray(truth))
rng = jax.random.key(0)

chain = np.empty((N_STEPS, 3))
n_acc = 0
for i in range(N_STEPS):
    rng, key = jax.random.split(rng)
    st, info = kernel.step(key, st)
    chain[i] = np.asarray(st.position)
    n_acc += bool(info.is_accepted)
    if (i + 1) % 100 == 0:
        np.savez(OUT, chain=chain[:i + 1], truth=truth, bounds=BOUNDS, sigma=np.asarray(SIGMA))
        print(f"{i+1}/{N_STEPS} | acc {n_acc/(i+1):.3f} | "
              f"e={chain[i,0]:.5f} m1={chain[i,1]:.4f} m2={chain[i,2]:.4f}", flush=True)

post = chain[N_STEPS // 5:]
print("\ntruth :", truth)
print("mean  :", post.mean(0))
print("std   :", post.std(0))
print("e 90% :", np.percentile(post[:, 0], 90))
print("acceptance:", n_acc / N_STEPS)
print("saved:", OUT)
