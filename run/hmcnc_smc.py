"""HM Cnc recovery with blackjax adaptive tempered SMC, run to convergence.

Gradient-free (pyTDI is NumPy, so the likelihood is neither traceable nor
differentiable), population-based, and it returns a log-evidence.

Config via environment variables:
    N_PARTICLES   (default 200)
    NUM_MCMC      (default 5)
    TARGET_ESS    (default 0.5)
    INJ           (default sangria_hmcnc_10x.h5)

    qsub run/run_hmcnc_smc.pbs
"""

import json
import os
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
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
inj_path = os.environ.get("INJ", scratch_dir + "/sangria_hmcnc_10x.h5")
tag = os.path.splitext(os.path.basename(inj_path))[0]
smc_path = f"{scratch_dir}/{tag}_smc.npz"

N_PARTICLES = int(os.environ.get("N_PARTICLES", 200))
NUM_MCMC = int(os.environ.get("NUM_MCMC", 5))
TARGET_ESS = float(os.environ.get("TARGET_ESS", 0.5))
PROP_SIGMA = jnp.asarray([0.05, 0.07, 0.07])
PROGRESS_EVERY = 100          # likelihood calls between progress lines

XYZ = ("X", "Y", "Z")
AET = ("A", "E", "T")
ll_chan = ("A", "E")
egb_pytdi_gen, measurement_order, delay_order = 1, 3, 3
bounds = np.array([[0.0, 0.9], [0.1, 1.5], [0.1, 1.5]])

print("jax:", jax.default_backend(), jax.devices(), flush=True)
print("injection:", inj_path, flush=True)
print(f"N_PARTICLES={N_PARTICLES} NUM_MCMC={NUM_MCMC} TARGET_ESS={TARGET_ESS}", flush=True)

# ----------------------------------------------------------------- load

with h5py.File(inj_path, "r") as f:
    t = f["t"][:]
    noise_xyz = {"t": t, **{ch: f["noise"][ch][:] for ch in XYZ}}
    signal_xyz = {ch: f["eccentric_signal"][ch][:] for ch in XYZ}
    injected_xyz = {"t": t, **{ch: f["injected"][ch][:] for ch in XYZ}}
    dt = float(f.attrs["segment_dt_s"])
    source_params = json.loads(f.attrs["egb_source_params_json"])
    injection_scale = float(f.attrs["injection_scale"])

# the file's stated scale must match what's actually in the data, or the
# template silently cannot match (this bit us once already)
for ch in XYZ:
    ratio = np.max(np.abs(injected_xyz[ch] - noise_xyz[ch])) / np.max(np.abs(signal_xyz[ch]))
    assert abs(ratio - injection_scale) < 1e-6, (
        f"{ch}: injected-noise is {ratio:.3f}x eccentric_signal but "
        f"injection_scale attr says {injection_scale}")

print(f"samples: {t.size}  dt: {dt}  injection_scale: {injection_scale}", flush=True)
print("source_params:", json.dumps(source_params, indent=2), flush=True)

# ----------------------------------------------------------------- setup

state = state_from_lisaorbits(default_lisaorbits("equal"), t)
geometry = precompute_jax_link_geometry(state)
prebuilt = prebuild_michelson(state, generation=egb_pytdi_gen, delay_order=delay_order)

noise_aet = aet_from_xyz(noise_xyz)
injected_aet = aet_from_xyz(injected_xyz)

psd_aet = {}
for ch in AET:
    freq, psd_aet[ch] = welch(noise_aet[ch], fs=1.0 / dt, nperseg=8192, detrend="constant")


def psd_on_grid(fg, wf, wp):
    safe = wf > 0
    out = np.full_like(fg, np.inf)
    out[fg > 0] = np.exp(np.interp(np.log(fg[fg > 0]), np.log(wf[safe]), np.log(wp[safe])))
    return out


data_freq, data_fd, psd_grid = {}, {}, {}
for ch in ll_chan:
    data_freq[ch] = np.fft.rfftfreq(injected_aet[ch].size, d=dt)
    data_fd[ch] = np.fft.rfft(injected_aet[ch]) * dt
    psd_grid[ch] = psd_on_grid(data_freq[ch], freq, psd_aet[ch])

# the Welch PSD does not resolve below its own lowest bin; don't extrapolate there
i0 = int(np.searchsorted(data_freq["A"], freq[freq > 0].min()))
df = data_freq["A"][1] - data_freq["A"][0]
d_w = {ch: 4.0 * df * dt * data_fd[ch][i0:] / psd_grid[ch][i0:] for ch in ll_chan}
w_hh = {ch: 4.0 * df * dt**2 / psd_grid[ch][i0:] for ch in ll_chan}


def htemp(params):
    ecc_i, m1_i, m2_i = params
    src = EccentricBinaryParams(
        mean_motion=np.pi * source_params["f0_hz"],
        eccentricity=ecc_i, m1_solar=m1_i, m2_solar=m2_i,
        distance_m=source_params["distance_m"],
        beta=source_params["beta"], lambda_=source_params["lambda_"],
        psi=source_params["psi"], inclination=source_params["inclination"],
        phi0=source_params["phi0"], fdot=source_params["fdot"],
    )
    raw = eccentric_xyz_jax(
        state, src, geometry=geometry, prebuilt=prebuilt, batch_size=1,
        physics_mode="1pn_periastron", generation=egb_pytdi_gen,
        measurement_order=measurement_order, delay_order=delay_order,
    )
    xyz = {ch: injection_scale * np.real(np.asarray(raw[ch], dtype=np.complex128)) for ch in XYZ}
    return aet_from_xyz(xyz)


def log_likelihood(params):
    template_aet = htemp(params)
    logl = 0.0
    for ch in ll_chan:
        h = np.fft.rfft(template_aet[ch])[i0:]
        logl += np.vdot(h, d_w[ch]).real - 0.5 * np.sum((h.real**2 + h.imag**2) * w_hh[ch])
    return float(logl)


n_calls = 0
t_start = time.time()


def _loglike_np(theta):
    global n_calls
    p = np.asarray(theta, dtype=np.float64)
    # out of bounds -> 0.0, not -inf: the prior already returns -inf, and
    # 0 * (-inf) = nan would break tempering at temperature 0
    if np.any(p < bounds[:, 0]) or np.any(p > bounds[:, 1]):
        return np.float64(0.0)
    n_calls += 1
    if n_calls % PROGRESS_EVERY == 0:
        el = time.time() - t_start
        print(f"    ... {n_calls} calls, {el:.0f}s, {el/n_calls:.2f} s/call", flush=True)
    return np.float64(log_likelihood(p))


def loglikelihood_fn(theta):
    return jax.pure_callback(_loglike_np, jax.ShapeDtypeStruct((), jnp.float64),
                             theta, vmap_method="sequential")


def logprior_fn(theta):
    inside = jnp.all((theta >= jnp.asarray(bounds[:, 0])) & (theta <= jnp.asarray(bounds[:, 1])))
    return jnp.where(inside, 0.0, -jnp.inf)


theta_true = np.array([source_params["eccentricity"],
                       source_params["m1_solar"], source_params["m2_solar"]])
print("\nlogL at truth:", log_likelihood(theta_true),
      f"-> implied SNR {np.sqrt(2 * log_likelihood(theta_true)):.1f}", flush=True)

# ----------------------------------------------------------------- sample

rw = blackjax.additive_step_random_walk.build_kernel()
prop = blackjax.mcmc.random_walk.normal(PROP_SIGMA)


def mcmc_step_fn(rng_key, st, logdensity_fn):
    return rw(rng_key, st, logdensity_fn, prop)


smc = blackjax.adaptive_tempered_smc(
    logprior_fn, loglikelihood_fn,
    mcmc_step_fn=mcmc_step_fn,
    mcmc_init_fn=blackjax.additive_step_random_walk.init,
    mcmc_parameters={}, resampling_fn=resampling.systematic,
    target_ess=TARGET_ESS, num_mcmc_steps=NUM_MCMC)

rng = jax.random.key(0)
init = jnp.column_stack([
    jax.random.uniform(jax.random.key(i), (N_PARTICLES,),
                       minval=bounds[i, 0], maxval=bounds[i, 1])
    for i in range(3)])
smc_state = smc.init(init)

print(f"\nstarting SMC, checkpointing to {smc_path}", flush=True)
n_calls, t_start = 0, time.time()
log_z = 0.0

for stage in range(200):
    rng, key = jax.random.split(rng)
    smc_state, info = smc.step(key, smc_state)
    log_z += float(info.log_likelihood_increment)
    temperature = float(smc_state.tempering_param)
    particles = np.asarray(smc_state.particles)
    weights = np.asarray(smc_state.weights)

    np.savez(smc_path, particles=particles, weights=weights, log_z=log_z,
             temperature=temperature, stage=stage + 1, n_calls=n_calls,
             truth=theta_true, bounds=bounds, injection_scale=injection_scale,
             n_particles=N_PARTICLES, num_mcmc=NUM_MCMC, target_ess=TARGET_ESS)

    print(f"stage {stage+1} | T {temperature:.4f} | logZ {log_z:.1f} | "
          f"calls {n_calls} | {time.time()-t_start:.0f}s | "
          f"mean e={particles[:,0].mean():.4f} m1={particles[:,1].mean():.4f} "
          f"m2={particles[:,2].mean():.4f}", flush=True)

    if temperature >= 1.0:
        break

particles = np.asarray(smc_state.particles)
print("\n" + "=" * 60)
print("truth          :", theta_true)
print("posterior mean :", particles.mean(0))
print("posterior std  :", particles.std(0))
print(f"e  90% upper limit: {np.percentile(particles[:, 0], 90):.5f}")
print(f"e  95% upper limit: {np.percentile(particles[:, 0], 95):.5f}")
print(f"logZ = {log_z:.3f}")
print(f"{stage+1} stages, {n_calls} calls, {time.time()-t_start:.0f}s total")
print("saved:", smc_path)
