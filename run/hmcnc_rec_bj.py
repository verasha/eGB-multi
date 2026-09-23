#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np 
import h5py
import warnings

from egb_jax_eccentric import (
    EccentricBinaryParams,
    aet_from_xyz,
    eccentric_complex_strain,
    eccentric_links_jax,
    eccentric_xyz_jax,
    lisa_orbit,
    precompute_jax_link_geometry,
    prepare_xyz_from_links,
    default_lisaorbits, 
    state_from_lisaorbits
)
from egb_jax_eccentric.constants import PARSEC_M#, KILOPARSEC_M
from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic
import astropy.units as u
import json
import time
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")  # no display on a compute node; plt.show() becomes a no-op
import matplotlib.pyplot as plt
import dynesty
import jax
import jax.numpy as jnp
import blackjax


# In[ ]:


import os
work_dir='/home/svu/e1498138/localgit/eGB-multi'
scratch_dir='/scratch/e1498138/eGB-multi'
os.chdir(work_dir)

inj_path = scratch_dir + '/sangria_hmcnc_10x.h5'


# # Read injected file

# In[ ]:


XYZ = ("X", "Y", "Z")
AET = ("A", "E", "T")


# In[4]:


with h5py.File(inj_path, "r") as f:
    t = f["t"][:]
    noise_xyz = {"t": t, **{ch: f["noise"][ch][:] for ch in XYZ}}
    signal_xyz = {ch: f["eccentric_signal"][ch][:] for ch in XYZ}
    injected_xyz = {"t": t, **{ch: f["injected"][ch][:] for ch in XYZ}}

    dt = f.attrs["segment_dt_s"]
    sangria_metadata = json.loads(f.attrs["sangria_metadata_json"])
    source_params = json.loads(f.attrs["egb_source_params_json"])
    # 'eccentric_signal' is stored unscaled but 'injected' carries the scaled
    # signal, so the template must be scaled by this to match the data
    injection_scale = float(f.attrs["injection_scale"])

print("samples:", t.size, "dt:", dt)
print("injection_scale:", injection_scale)
print("source_params:", json.dumps(source_params, indent=2))


# In[5]:


state = state_from_lisaorbits(default_lisaorbits("equal"), t)
geometry = precompute_jax_link_geometry(state)

egb_pytdi_gen = 1
measurement_order = 3
delay_order = 3

# pyTDI's nested delays and Doppler factors depend only on the orbit, not on the
# source, so prepare them once and reuse for every likelihood call. Upstream's
# version also passes eta_ij = sci_ij directly (valid here since all reference
# and metrology channels are zero): ~4x on the TDI stage, ~2x on the full call.
xyz_eval = prepare_xyz_from_links(
    state,
    generation=egb_pytdi_gen,
    measurement_order=measurement_order,
    delay_order=delay_order,
)


# # Rotate to AET & PSD Estimate

# In[ ]:


noise_aet = aet_from_xyz(noise_xyz)
signal_aet = aet_from_xyz(signal_xyz)
injected_aet = aet_from_xyz(injected_xyz)

print("noise AET max abs:", {ch: float(np.max(np.abs(noise_aet[ch]))) for ch in AET})
print("signal AET max abs:", {ch: float(np.max(np.abs(signal_aet[ch]))) for ch in AET})


# In[7]:


fs = 1.0 / dt
nperseg = 8192 #like in the paper but do note its for dt=10s
# get psd on same grid
# nperseg = noise_xyz["X"].size 

psd_xyz, psd_aet = {}, {}
for ch in XYZ:
    freq, psd_xyz[ch] = welch(noise_xyz[ch], fs=fs, nperseg=nperseg, detrend="constant")
for ch in AET:
    _, psd_aet[ch] = welch(noise_aet[ch], fs=fs, nperseg=nperseg, detrend="constant")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for ch in XYZ:
    axes[0].loglog(freq[1:], psd_xyz[ch][1:], label=ch)
axes[0].set_title("X/Y/Z")

for ch in AET:
    axes[1].loglog(freq[1:], psd_aet[ch][1:], label=ch)
axes[1].set_title("A/E/T")

for ax in axes:
    ax.set_xlabel("frequency [Hz]")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
axes[0].set_ylabel("Welch PSD")

fig.tight_layout()
plt.show()


# # FFT of injected data

# In[ ]:


def fdom(x, dt):
    n = x.size
    xf = np.fft.rfft(x) * dt
    freq = np.fft.rfftfreq(n, d=dt)
    return freq, xf

ll_chan = ("A", "E") #exclude T cuz we hate it C:


# In[ ]:


data_freq, data_fd = {}, {}
for ch in ll_chan:
    data_freq_i, data_fd_i = fdom(injected_aet[ch], dt)
    data_freq[ch] = data_freq_i
    data_fd[ch] = data_fd_i


# In[10]:


for ch in ll_chan:
    plt.loglog(data_freq[ch][1:], np.abs(data_fd[ch][1:]), label=ch)
plt.xlabel("frequency [Hz]")
plt.ylabel("Fourier amplitude")
plt.grid(alpha=0.3, which="both")
plt.title("Fourier transform of injected A/E channels")
plt.legend()
plt.show()


# In[11]:


for ch in ll_chan:
    freq_s, xf = fdom(signal_aet[ch], dt)
    plt.loglog(freq_s[1:], np.abs(xf[1:]), label=ch)
plt.xlabel("frequency [Hz]")
plt.ylabel("Fourier amplitude")
plt.grid(alpha=0.3, which="both")
plt.legend()
plt.title("Fourier amplitude of injected signal")
plt.show()


# # Interpolate PSD on the same grid

# In[12]:


def psd_on_grid(freq_grid, welch_freq, welch_psd):
    safe = welch_freq > 0
    log_psd = np.interp(np.log(freq_grid[freq_grid > 0]),
                         np.log(welch_freq[safe]), np.log(welch_psd[safe]))
    out = np.full_like(freq_grid, np.inf)
    out[freq_grid > 0] = np.exp(log_psd)
    return out

psd_grid = {}
for ch in ll_chan:
    psd_grid[ch] = psd_on_grid(data_freq[ch], freq, psd_aet[ch])
    plt.loglog(data_freq[ch][1:], psd_grid[ch][1:], label=ch)

plt.xlabel("frequency [Hz]")
plt.ylabel("PSD")
plt.grid(alpha=0.3, which="both")
plt.title("Interpolated PSD")
plt.legend()
plt.show()


# # Waveform template func

# In[13]:


def htemp(params):
    # params to vary
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

    # waveform on the six one-way links (JAX/GPU), then the prepared pyTDI
    # operators (NumPy) -- interpolation order is baked into xyz_eval
    links = eccentric_links_jax(
        source_temp,
        geometry,
        batch_size=1, #NOTE: think abt different sampler that is able to handle batched pool evaluations
        physics_mode="1pn_periastron", #NOTE: 1pn
    )
    raw_temp_xyz = xyz_eval(links)

    # scale to match the injected data, which carries injection_scale * signal
    xyz = {ch: injection_scale * np.real(np.asarray(raw_temp_xyz[ch], dtype=np.complex128)) for ch in XYZ}
    return aet_from_xyz(xyz)


# # Log likelihood

# In[ ]:


# take freq min of the welch psd 
# its abt 10^-5 meanwhile injected data goes to 10^-7..

freq_min = freq[freq > 0].min()
freq_mask = data_freq["A"] >= freq_min

df = data_freq["A"][1] - data_freq["A"][0]


# expand $-\frac12\langle d-h|d-h\rangle = -\frac12 \langle d|d\rangle + \langle d|h \rangle - \frac12 \langle h|h \rangle$

# In[15]:


# def log_likelihood(params):
#     template_aet = htemp(params)
#     logl = 0.0
#     for ch in ("A", "E"):
#         _, h_fd = fdom(template_aet[ch], dt)
#         d = data_fd[ch][freq_mask]
#         h = h_fd[freq_mask]
#         sn = psd_grid[ch][freq_mask]
#         dh = 4.0 * np.sum(np.real(d * np.conj(h)) / sn) * df
#         hh = 4.0 * np.sum(np.real(h * np.conj(h)) / sn) * df
#         logl += dh - 0.5 * hh
#     return float(logl)


# vectorized 

# In[16]:


# --- one-off setup, next to freq_mask/df ---
i0 = int(np.searchsorted(data_freq["A"], freq_min))  

# fold all constants (4, df, dt, 1/Sn) into precomputed weights
w_hh = {ch: 4.0 * df * dt**2 / psd_grid[ch][i0:] for ch in ll_chan}
d_w  = {ch: 4.0 * df * dt * data_fd[ch][i0:] / psd_grid[ch][i0:] for ch in ll_chan}


def log_likelihood(params):
    template_aet = htemp(params)
    logl = 0.0
    for ch in ll_chan:
        h = np.fft.rfft(template_aet[ch])[i0:]        # raw rfft; dt folded into weights above
        dh = np.vdot(h, d_w[ch]).real                 # sum(conj(h) * 4 df dt d / Sn)
        hh = np.sum((h.real**2 + h.imag**2) * w_hh[ch])
        logl += dh - 0.5 * hh
    return float(logl)


# # Priors

# In[17]:


def prior_transform(u):
    ecc_u, m1_u, m2_u = u
    eccentricity = ecc_u * 0.9          # uniform prior: e in [0, 0.9)
    m1_solar = 0.1 + m1_u * (1.5 - 0.1)  # uniform prior: m1 in [0.1, 1.5] Msun
    m2_solar = 0.1 + m2_u * (1.5 - 0.1)  # same range for m2
    return eccentricity, m1_solar, m2_solar


# In[18]:


bounds =  np.array([[0.0, 0.9], [0.1, 1.5], [0.1, 1.5]])
def logdensity_fn(theta):
    params = np.asarray(theta, dtype=np.float64)
    # uniform priors
    if np.any(params < bounds[:, 0]) or np.any(params > bounds[:, 1]):
        return jnp.asarray(-np.inf)
    return jnp.asarray(log_likelihood(params))


# # Sample try

# In[19]:


theta_true = np.array(
    [source_params["eccentricity"], source_params["m1_solar"], source_params["m2_solar"]]
)


# In[20]:


import time
t0 = time.time()
ll_true = logdensity_fn(theta_true)
print("log likelihood at true params:", ll_true)
print("one call takes:", time.time() - t0, "seconds")


# In[21]:


t0 = time.time()
ll_arb=logdensity_fn([0.5, 0.8, 0.9])
print("log likelihood at arbitrary params:", ll_arb)
print("second call takes:", time.time() - t0, "seconds")


# In[22]:


t0 = time.time()
print("log likelihood at arbitrary params:", logdensity_fn([0.2, 0.3, 0.5]))
print(" call takes:", time.time() - t0, "seconds")


# # BJBJBJBJ

# In[ ]:


import blackjax.smc.resampling as resampling

N_PARTICLES = 50        # pilot size
N_PROD = 200            # production size, only used to project the walltime
NUM_MCMC = 5
TARGET_ESS = 0.5
PROP_SIGMA = jnp.asarray([0.05, 0.07, 0.07])
smc_path = scratch_dir + "/hmcnc_smc_pilot.npz"

n_calls = 0

def _loglike_np(theta):
    global n_calls
    p = np.asarray(theta, dtype=np.float64)
    # out of bounds: return 0.0, NOT -inf. The prior already gives -inf there,
    # and 0 * (-inf) = nan would break the tempering at temperature 0.
    if np.any(p < bounds[:, 0]) or np.any(p > bounds[:, 1]):
        return np.float64(0.0)
    n_calls += 1
    return np.float64(log_likelihood(p))

def loglikelihood_fn(theta):
    return jax.pure_callback(_loglike_np, jax.ShapeDtypeStruct((), jnp.float64),
                             theta, vmap_method="sequential")

def logprior_fn(theta):
    inside = jnp.all((theta >= jnp.asarray(bounds[:, 0])) & (theta <= jnp.asarray(bounds[:, 1])))
    return jnp.where(inside, 0.0, -jnp.inf)

print("_loglike_np at truth:", _loglike_np(theta_true), "(expect ~11800)", flush=True)

rw = blackjax.additive_step_random_walk.build_kernel()
prop = blackjax.mcmc.random_walk.normal(PROP_SIGMA)
def mcmc_step_fn(rng_key, rw_state, logdensity_fn):
    return rw(rng_key, rw_state, logdensity_fn, prop)

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

# NOTE: smc_state, not `state` -- `state` is the LISA orbit used by htemp
smc_state = smc.init(init)

log_z = 0.0
n_calls = 0
t_start = time.time()

for stage in range(60):
    rng, k = jax.random.split(rng)
    smc_state, info = smc.step(k, smc_state)
    log_z += float(info.log_likelihood_increment)
    T = float(smc_state.tempering_param)
    p = np.asarray(smc_state.particles)
    elapsed = time.time() - t_start

    np.savez(smc_path, particles=p, weights=np.asarray(smc_state.weights),
             log_z=log_z, temperature=T, stage=stage + 1, n_calls=n_calls,
             truth=theta_true, bounds=bounds, injection_scale=injection_scale,
             n_particles=N_PARTICLES, num_mcmc=NUM_MCMC, target_ess=TARGET_ESS,
             elapsed_s=elapsed)

    print(f"stage {stage+1} | T {T:.4f} | logZ {log_z:.1f} | calls {n_calls} | "
          f"{elapsed:.0f}s | {elapsed/max(n_calls,1):.2f} s/call | "
          f"mean e={p[:,0].mean():.4f} m1={p[:,1].mean():.4f} m2={p[:,2].mean():.4f}",
          flush=True)
    if T >= 1.0:
        break

elapsed = time.time() - t_start
n_stages = stage + 1

print("\n" + "=" * 62)
print("truth         :", theta_true)
print("posterior mean:", p.mean(0))
print("posterior std :", p.std(0))
print(f"e 90% upper limit: {np.percentile(p[:, 0], 90):.5f}")
print(f"e 95% upper limit: {np.percentile(p[:, 0], 95):.5f}")
print(f"logZ = {log_z:.2f}")

# ---- walltime projection for the production run -------------------------
s_per_call = elapsed / max(n_calls, 1)
calls_per_stage = n_calls / n_stages
print("\n--- walltime sizing ---")
print(f"pilot: {n_stages} stages, {n_calls} calls, {elapsed/60:.1f} min "
      f"({s_per_call:.2f} s/call, {calls_per_stage:.0f} calls/stage)")

# calls scale with particle count; stage count is set by target_ess, so assume
# it carries over (pad it, since a sharper posterior can need more stages)
for stages_assumed in (n_stages, n_stages + 4):
    prod_calls = calls_per_stage * (N_PROD / N_PARTICLES) * stages_assumed
    prod_hours = prod_calls * s_per_call / 3600
    print(f"  {N_PROD} particles, {stages_assumed} stages -> "
          f"{prod_calls:.0f} calls, {prod_hours:.1f} h")

pad = 1.5
worst_h = calls_per_stage * (N_PROD / N_PARTICLES) * (n_stages + 4) * s_per_call / 3600
print(f"\nsuggested PBS walltime ({pad:g}x margin): {int(np.ceil(worst_h * pad))}:00:00")


# In[ ]:





# In[ ]:




