"""Where does a single log_likelihood call actually spend its time, on the
current (prebuilt-Michelson) path?

Breaks the call into its three stages and times each, then gives a cProfile
breakdown for the dominant one.

    qsub -I -q interactive_gpu -l select=1:ncpus=4:ngpus=1 -l walltime=1:00:00
    module load cuda12.4/toolkit/12.4.1
    source /scratch/e1498138/anaconda3/bin/activate egb-multi
    python /home/svu/e1498138/localgit/eGB-multi/run/profile_stages.py
"""

import cProfile
import io
import json
import pstats
import time

import h5py
import numpy as np
from scipy.signal import welch

import jax

from egb_jax_eccentric import (
    EccentricBinaryParams,
    aet_from_xyz,
    default_lisaorbits,
    eccentric_links_jax,
    prebuild_michelson,
    precompute_jax_link_geometry,
    state_from_lisaorbits,
    xyz_from_prebuilt,
)

print("jax backend:", jax.default_backend(), jax.devices())

inj_path = "/scratch/e1498138/eGB-multi/sangria_hmcnc_true.h5"
XYZ = ("X", "Y", "Z")
ll_chan = ("A", "E")
egb_pytdi_gen, measurement_order, delay_order = 1, 3, 3

with h5py.File(inj_path, "r") as f:
    t = f["t"][:]
    noise_xyz = {"t": t, **{ch: f["noise"][ch][:] for ch in XYZ}}
    injected_xyz = {"t": t, **{ch: f["injected"][ch][:] for ch in XYZ}}
    dt = float(f.attrs["segment_dt_s"])
    source_params = json.loads(f.attrs["egb_source_params_json"])

print("samples:", t.size, "dt:", dt)

state = state_from_lisaorbits(default_lisaorbits("equal"), t)
geometry = precompute_jax_link_geometry(state)
prebuilt = prebuild_michelson(state, generation=egb_pytdi_gen, delay_order=delay_order)

noise_aet = aet_from_xyz(noise_xyz)
injected_aet = aet_from_xyz(injected_xyz)

psd_aet = {}
for ch in ("A", "E", "T"):
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

i0 = int(np.searchsorted(data_freq["A"], freq[freq > 0].min()))
df = data_freq["A"][1] - data_freq["A"][0]
d_w = {ch: 4.0 * df * dt * data_fd[ch][i0:] / psd_grid[ch][i0:] for ch in ll_chan}
w_hh = {ch: 4.0 * df * dt**2 / psd_grid[ch][i0:] for ch in ll_chan}

source = EccentricBinaryParams(
    mean_motion=np.pi * source_params["f0_hz"],
    eccentricity=source_params["eccentricity"],
    m1_solar=source_params["m1_solar"],
    m2_solar=source_params["m2_solar"],
    distance_m=source_params["distance_m"],
    beta=source_params["beta"],
    lambda_=source_params["lambda_"],
    psi=source_params["psi"],
    inclination=source_params["inclination"],
    phi0=source_params["phi0"],
    fdot=source_params["fdot"],
)


# ---- the three stages of one likelihood call -------------------------------

def stage_links():
    """JAX/GPU: eccentric waveform on the six one-way links."""
    return eccentric_links_jax(
        source, geometry, batch_size=1, physics_mode="1pn_periastron"
    )


def stage_tdi(links):
    """pyTDI/NumPy: apply the prebuilt Michelson operators -> XYZ -> AET."""
    raw = xyz_from_prebuilt(prebuilt, links, measurement_order=measurement_order)
    return aet_from_xyz({ch: np.real(np.asarray(raw[ch], dtype=np.complex128)) for ch in XYZ})


def stage_inner(template_aet):
    """FFT + noise-weighted inner products."""
    logl = 0.0
    for ch in ll_chan:
        h = np.fft.rfft(template_aet[ch])[i0:]
        logl += np.vdot(h, d_w[ch]).real - 0.5 * np.sum((h.real**2 + h.imag**2) * w_hh[ch])
    return float(logl)


def timeit(fn, n=3):
    fn()
    t0 = time.time()
    for _ in range(n):
        out = fn()
    return (time.time() - t0) / n, out


print("\n--- per-stage timing (3 runs each, after warmup) ---")
t_links, links = timeit(stage_links)
t_tdi, template_aet = timeit(lambda: stage_tdi(links))
t_inner, logl = timeit(lambda: stage_inner(template_aet))
total = t_links + t_tdi + t_inner

print(f"  1. links   (JAX/GPU waveform)      {t_links:7.3f} s   {100 * t_links / total:5.1f}%")
print(f"  2. TDI     (pyTDI/NumPy, prebuilt) {t_tdi:7.3f} s   {100 * t_tdi / total:5.1f}%")
print(f"  3. inner   (FFT + inner products)  {t_inner:7.3f} s   {100 * t_inner / total:5.1f}%")
print(f"  {'total':-<36} {total:7.3f} s")
print(f"  (logL = {logl:.4f})")

print("\n--- cProfile of the dominant stage (TDI) ---")
profiler = cProfile.Profile()
profiler.enable()
stage_tdi(links)
profiler.disable()
stream = io.StringIO()
pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(20)
print(stream.getvalue())
