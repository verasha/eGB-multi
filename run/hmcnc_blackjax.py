"""HM Cnc recovery with blackjax random-walk Metropolis (gradient-free).

Why gradient-free: the likelihood goes through pyTDI, which is pure NumPy, so it
is neither jax-traceable nor differentiable. NUTS/HMC are unavailable until the
TDI step is reimplemented in JAX.

Why no jit / lax.scan: each log-density evaluation is ~1.1 s of NumPy, so a
compiled loop buys nothing. Running the kernel eagerly also avoids needing
jax.pure_callback, and lets us print progress and checkpoint.
"""

import json
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import blackjax
from scipy.signal import welch

from egb_jax_eccentric import (
    EccentricBinaryParams,
    aet_from_xyz,
    default_lisaorbits,
    eccentric_xyz_jax,
    prebuild_michelson,
    precompute_jax_link_geometry,
    state_from_lisaorbits,
)

scratch_dir = "/scratch/e1498138/eGB-multi"
inj_path = scratch_dir + "/sangria_hmcnc_true.h5"
chain_path = scratch_dir + "/hmcnc_blackjax_chain.npz"

XYZ = ("X", "Y", "Z")
AET = ("A", "E", "T")
ll_chan = ("A", "E")

egb_pytdi_gen = 1
measurement_order = 3
delay_order = 3

N_STEPS = 20000
CHECKPOINT_EVERY = 50
# RWM proposal width per parameter (ecc, m1, m2). Tune so acceptance lands ~0.2-0.3.
SIGMA = np.array([0.01, 0.03, 0.03])

# Uniform prior bounds, matching the dynesty prior_transform
BOUNDS = np.array([[0.0, 0.9], [0.1, 1.5], [0.1, 1.5]])

# ----------------------------------------------------------------- setup

with h5py.File(inj_path, "r") as f:
    t = f["t"][:]
    noise_xyz = {"t": t, **{ch: f["noise"][ch][:] for ch in XYZ}}
    injected_xyz = {"t": t, **{ch: f["injected"][ch][:] for ch in XYZ}}
    dt = float(f.attrs["segment_dt_s"])
    source_params = json.loads(f.attrs["egb_source_params_json"])

print("samples:", t.size, "dt:", dt, flush=True)

state = state_from_lisaorbits(default_lisaorbits("equal"), t)
geometry = precompute_jax_link_geometry(state)
prebuilt = prebuild_michelson(state, generation=egb_pytdi_gen, delay_order=delay_order)

noise_aet = aet_from_xyz(noise_xyz)
injected_aet = aet_from_xyz(injected_xyz)

fs = 1.0 / dt
nperseg = 8192
psd_aet = {}
for ch in AET:
    freq, psd_aet[ch] = welch(noise_aet[ch], fs=fs, nperseg=nperseg, detrend="constant")


def fdom(x, dt):
    return np.fft.rfftfreq(x.size, d=dt), np.fft.rfft(x) * dt


def psd_on_grid(freq_grid, welch_freq, welch_psd):
    safe = welch_freq > 0
    log_psd = np.interp(
        np.log(freq_grid[freq_grid > 0]), np.log(welch_freq[safe]), np.log(welch_psd[safe])
    )
    out = np.full_like(freq_grid, np.inf)
    out[freq_grid > 0] = np.exp(log_psd)
    return out


data_freq, data_fd, psd_grid = {}, {}, {}
for ch in ll_chan:
    data_freq[ch], data_fd[ch] = fdom(injected_aet[ch], dt)
    psd_grid[ch] = psd_on_grid(data_freq[ch], freq, psd_aet[ch])

freq_min = freq[freq > 0].min()
i0 = int(np.searchsorted(data_freq["A"], freq_min))
df = data_freq["A"][1] - data_freq["A"][0]

# constants folded in once: 4 df dt / Sn and 4 df dt^2 / Sn
d_w = {ch: 4.0 * df * dt * data_fd[ch][i0:] / psd_grid[ch][i0:] for ch in ll_chan}
w_hh = {ch: 4.0 * df * dt**2 / psd_grid[ch][i0:] for ch in ll_chan}


def htemp(params):
    ecc_i, m1_i, m2_i = params
    source_temp = EccentricBinaryParams(
        mean_motion=np.pi * source_params["f0_hz"],
        eccentricity=ecc_i,
        m1_solar=m1_i,
        m2_solar=m2_i,
        distance_m=source_params["distance_m"],
        beta=source_params["beta"],
        lambda_=source_params["lambda_"],
        psi=source_params["psi"],
        inclination=source_params["inclination"],
        phi0=source_params["phi0"],
        fdot=source_params["fdot"],
    )
    raw = eccentric_xyz_jax(
        state,
        source_temp,
        geometry=geometry,
        prebuilt=prebuilt,
        batch_size=1,
        physics_mode="1pn_periastron",
        generation=egb_pytdi_gen,
        measurement_order=measurement_order,
        delay_order=delay_order,
    )
    xyz = {ch: np.real(np.asarray(raw[ch], dtype=np.complex128)) for ch in XYZ}
    return aet_from_xyz(xyz)


def log_likelihood(params):
    template_aet = htemp(params)
    logl = 0.0
    for ch in ll_chan:
        h = np.fft.rfft(template_aet[ch])[i0:]
        dh = np.vdot(h, d_w[ch]).real
        hh = np.sum((h.real**2 + h.imag**2) * w_hh[ch])
        logl += dh - 0.5 * hh
    return float(logl)


n_calls = 0


def logdensity_fn(theta):
    """log prior + log likelihood, evaluated eagerly on concrete values."""
    global n_calls
    params = np.asarray(theta, dtype=np.float64)
    # uniform priors: reject outside the box without paying for a waveform
    if np.any(params < BOUNDS[:, 0]) or np.any(params > BOUNDS[:, 1]):
        return jnp.asarray(-np.inf)
    n_calls += 1
    return jnp.asarray(log_likelihood(params))


# ----------------------------------------------------------------- sample

theta_true = np.array(
    [source_params["eccentricity"], source_params["m1_solar"], source_params["m2_solar"]]
)

print("\nsanity check:", flush=True)
print("  logdensity at truth :", float(logdensity_fn(theta_true)), flush=True)
print("  logdensity at e=0.5 :", float(logdensity_fn(np.array([0.5, theta_true[1], theta_true[2]]))), flush=True)

rng = jax.random.key(0)
kernel = blackjax.additive_step_random_walk.normal_random_walk(
    logdensity_fn, sigma=jnp.asarray(SIGMA)
)

# start at the truth: with ~1.1 s/step a blind burn-in is unaffordable, and the
# question here is what the posterior looks like, not whether a blind search finds it
position = jnp.asarray(theta_true)
sampler_state = kernel.init(position)

chain = np.empty((N_STEPS, 3))
logdens = np.empty(N_STEPS)
n_accept = 0
t_start = time.time()

for i in range(N_STEPS):
    rng, subkey = jax.random.split(rng)
    sampler_state, info = kernel.step(subkey, sampler_state)

    chain[i] = np.asarray(sampler_state.position)
    logdens[i] = float(sampler_state.logdensity)
    n_accept += bool(info.is_accepted)

    if (i + 1) % CHECKPOINT_EVERY == 0:
        elapsed = time.time() - t_start
        np.savez(
            chain_path,
            chain=chain[: i + 1],
            logdensity=logdens[: i + 1],
            sigma=SIGMA,
            bounds=BOUNDS,
            truth=theta_true,
            n_steps_done=i + 1,
            acceptance=n_accept / (i + 1),
        )
        print(
            f"step {i + 1}/{N_STEPS} | acc {n_accept / (i + 1):.3f} | "
            f"logp {logdens[i]:.2f} | {elapsed / (i + 1):.2f} s/step | "
            f"e={chain[i, 0]:.4f} m1={chain[i, 1]:.4f} m2={chain[i, 2]:.4f}",
            flush=True,
        )

print(f"\ndone: {N_STEPS} steps, {n_calls} likelihood calls, "
      f"acceptance {n_accept / N_STEPS:.3f}, {time.time() - t_start:.0f} s")
print("chain saved to", chain_path)
