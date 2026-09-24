"""Fisher-matrix forecast for the HM Cnc eccentric recovery.

Gamma_ij = (dh/dtheta_i | dh/dtheta_j) using the same noise-weighted inner
product as the MCMC likelihood, so the two are directly comparable. The
covariance estimate is C = Gamma^-1.

Fisher needs only the PSD and the waveform derivatives -- not the injected
data -- so a whole eccentricity scan costs 7 waveform evaluations per point
instead of one 30k-step chain. One injection file is read purely to supply the
time grid, the noise realisation for the PSD, and the fixed source parameters.

    python run/hmcnc_fisher.py                    # scan e, default grid
    ECC_SCAN=0.0,0.1,0.3 python run/hmcnc_fisher.py
    STEP_CHECK=1 python run/hmcnc_fisher.py       # step-size convergence only

Caveats this script does not paper over:
  * Fisher is a high-SNR linearisation. It is known to overestimate precision
    for galactic binaries at modest SNR (Vallisneri 2008). Cross-check against
    the e=0.3 MCMC, which is the one chain that is not prior-limited.
  * At e=0 the mass block is numerically singular -- that IS the degeneracy.
    BEWARE: pinv does not blow the sigma up, it silently DROPS the null
    direction and returns an absurdly SMALL sigma for a rank-deficient block
    (e=0 reports sigma(m1)~7e-4 and rho=+1.000, both artefacts). Any row
    flagged SINGULAR below must be read as "unconstrained", never as its
    printed sigma.
  * Eccentric harmonic amplitudes scale as e^|n-2|, so d/de is delicate near
    e=0; a central difference would also straddle e<0. A second-order forward
    difference is used whenever e_fid - eps < 0.
"""

import json
import os

import h5py
import numpy as np
from scipy.signal import welch

from egb_jax_eccentric import (
    EccentricBinaryParams, aet_from_xyz, default_lisaorbits, eccentric_links_jax,
    precompute_jax_link_geometry, prepare_xyz_from_links, state_from_lisaorbits,
)

SCRATCH = "/scratch/e1498138/eGB-multi"
INJ = os.environ.get("INJ", f"{SCRATCH}/sangria_hmcnc_10x_e0_3.h5")
OUT = os.environ.get("OUT", f"{SCRATCH}/hmcnc_fisher.npz")
STEP_CHECK = bool(int(os.environ.get("STEP_CHECK", 0)))
ECC_SCAN = [float(x) for x in os.environ.get(
    "ECC_SCAN", "0.0,0.02,0.05,0.08,0.10,0.13,0.16,0.20,0.25,0.30,0.35,0.40").split(",")]

XYZ, CH = ("X", "Y", "Z"), ("A", "E")
NAMES = ["e", "m1", "m2"]
NDIM = 3
# Absolute finite-difference steps. Validate with STEP_CHECK=1 before trusting.
EPS = np.array([float(os.environ.get("EPS_E", 2e-4)),
                float(os.environ.get("EPS_M1", 2e-5)),
                float(os.environ.get("EPS_M2", 2e-5))])

print(f"INJ={INJ}\nEPS={EPS}", flush=True)

# --- data: time grid, noise realisation (PSD only), fixed source parameters
DURATION_DAYS = float(os.environ.get("DURATION_DAYS", 0))   # 0 = use whole file
SCALE_OVERRIDE = float(os.environ.get("SCALE_OVERRIDE", 0))  # 0 = use file's scale

with h5py.File(INJ, "r") as f:
    dt = float(f.attrs["segment_dt_s"])
    n = f["t"].shape[0]
    if DURATION_DAYS > 0:
        n = min(n, int(round(DURATION_DAYS * 86400.0 / dt)))
    t = f["t"][:n]
    noise = aet_from_xyz({c: f["noise"][c][:n] for c in XYZ})
    sp = json.loads(f.attrs["egb_source_params_json"])
    scale = SCALE_OVERRIDE if SCALE_OVERRIDE > 0 else float(f.attrs["injection_scale"])
print(f"N={n}  T={n*dt/86400:.2f}d  scale={scale}", flush=True)

state = state_from_lisaorbits(default_lisaorbits("equal"), t)
geom = precompute_jax_link_geometry(state)
xyz_eval = prepare_xyz_from_links(state, generation=1, measurement_order=3, delay_order=3)

psd = {}
for c in ("A", "E", "T"):
    freq, p = welch(noise[c], fs=1 / dt, nperseg=8192, detrend="constant")
    psd[c] = p

fgrid = np.fft.rfftfreq(t.size, d=dt)
i0 = int(np.searchsorted(fgrid, freq[freq > 0].min()))
df = fgrid[1] - fgrid[0]
# Identical weighting to the likelihood: (a|b) = sum (a.re b.re + a.im b.im) * w
w = {}
for c in CH:
    sn = np.exp(np.interp(np.log(fgrid[i0:]), np.log(freq[1:]), np.log(psd[c][1:])))
    w[c] = 4 * df * dt**2 / sn


def template(p):
    """Frequency-domain A,E template at theta = (e, m1, m2), matching the data scale."""
    src = EccentricBinaryParams(
        mean_motion=np.pi * sp["f0_hz"], eccentricity=p[0], m1_solar=p[1], m2_solar=p[2],
        distance_m=sp["distance_m"], beta=sp["beta"], lambda_=sp["lambda_"],
        psi=sp["psi"], inclination=sp["inclination"], phi0=sp["phi0"], fdot=sp["fdot"])
    xyz = xyz_eval(eccentric_links_jax(src, geom, batch_size=1,
                                       physics_mode="1pn_periastron"))
    aet = aet_from_xyz({c: scale * np.real(np.asarray(xyz[c], dtype=np.complex128))
                        for c in XYZ})
    return {c: np.fft.rfft(aet[c])[i0:] for c in CH}


def inner(a, b):
    return sum(np.sum((a[c].real * b[c].real + a[c].imag * b[c].imag) * w[c]) for c in CH)


def derivative(p, i, eps):
    """d(template)/d(theta_i). Forward-differenced if a central step goes negative."""
    if p[i] - eps < 0.0:
        # second-order forward: (-3 f0 + 4 f1 - f2) / (2 eps)
        f0, f1, f2 = (template(p), ) + tuple(
            template(np.where(np.arange(NDIM) == i, p + k * eps, p)) for k in (1, 2))
        return {c: (-3 * f0[c] + 4 * f1[c] - f2[c]) / (2 * eps) for c in CH}
    pp, pm = p.copy(), p.copy()
    pp[i] += eps
    pm[i] -= eps
    fp, fm = template(pp), template(pm)
    return {c: (fp[c] - fm[c]) / (2 * eps) for c in CH}


def fisher(p, eps):
    d = [derivative(p, i, eps[i]) for i in range(NDIM)]
    G = np.zeros((NDIM, NDIM))
    for i in range(NDIM):
        for j in range(i, NDIM):
            G[i, j] = G[j, i] = inner(d[i], d[j])
    return G


def derived_sigma(p, C):
    """Propagate C into sigma(Mc) and sigma(M) with the analytic Jacobians."""
    m1, m2 = p[1], p[2]
    M = m1 + m2
    Mc = (m1 * m2) ** 0.6 / M ** 0.2
    jM = np.array([0.0, 1.0, 1.0])
    jMc = np.array([0.0, Mc * (0.6 / m1 - 0.2 / M), Mc * (0.6 / m2 - 0.2 / M)])
    return float(np.sqrt(jMc @ C @ jMc)), float(np.sqrt(jM @ C @ jM)), Mc, M


# --- step-size convergence: Gamma_ii should plateau over a decade or two
if STEP_CHECK:
    p0 = np.array([0.3, sp["m1_solar"], sp["m2_solar"]])
    print("\nstep-size check at e=0.3 -- look for a plateau in each column")
    print(f"{'factor':>8s} " + " ".join(f"{'G_' + n + n:>14s}" for n in NAMES))
    for fac in (100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03):
        e = EPS * fac
        d = [derivative(p0, i, e[i]) for i in range(NDIM)]
        row = [inner(d[i], d[i]) for i in range(NDIM)]
        print(f"{fac:8.2f} " + " ".join(f"{v:14.6e}" for v in row), flush=True)
    raise SystemExit(0)

# --- eccentricity scan
print(f"\n{'e':>6s} {'SNR':>7s} | {'sig(e)':>9s} {'sig(m1)':>9s} {'sig(m2)':>9s} "
      f"{'rho':>7s} | {'sig(Mc)':>9s} {'sig(M)':>9s} | {'cond':>9s}")
rows = []
for ecc in ECC_SCAN:
    p = np.array([ecc, sp["m1_solar"], sp["m2_solar"]])
    G = fisher(p, EPS)
    snr = float(np.sqrt(inner(template(p), template(p))))
    # pinv, not inv: at low e the mass block is near-singular by construction.
    C = np.linalg.pinv(G)
    sig = np.sqrt(np.diag(C))
    rho = C[1, 2] / (sig[1] * sig[2])
    sMc, sM, _, _ = derived_sigma(p, C)
    cond = np.linalg.cond(G[1:, 1:])
    # cond above ~1/sqrt(eps_machine) means pinv has truncated a direction and
    # the mass sigmas below are projections, not uncertainties.
    flag = "  <-- SINGULAR: masses unconstrained, sigmas meaningless" if cond > 1e8 else ""
    print(f"{ecc:6.3f} {snr:7.1f} | {sig[0]:9.2e} {sig[1]:9.2e} {sig[2]:9.2e} "
          f"{rho:+7.3f} | {sMc:9.2e} {sM:9.2e} | {cond:9.2e}{flag}", flush=True)
    rows.append((ecc, snr, *sig, rho, sMc, sM, cond, *G.ravel()))

arr = np.array(rows)
np.savez(OUT, scan=arr, ecc=arr[:, 0], snr=arr[:, 1], sigma=arr[:, 2:5],
         rho=arr[:, 5], sigma_Mc=arr[:, 6], sigma_M=arr[:, 7], cond=arr[:, 8],
         gamma=arr[:, 9:].reshape(-1, NDIM, NDIM), eps=EPS, names=NAMES)
print("\nsaved:", OUT)

# --- cross-check against the one MCMC chain that is not prior-limited
# Must name a chain for THIS system -- comparing across systems is meaningless.
chain_path = os.environ.get("CHECK_CHAIN", "")
if chain_path and os.path.exists(chain_path):
    c = np.load(chain_path)["chain"]
    c = c[len(c) // 5:]
    k = int(np.argmin(np.abs(arr[:, 0] - 0.3)))
    print(f"\ncross-check at e={arr[k,0]:.2f}  (MCMC e=0.3 chain, tight prior)")
    print(f"{'':10s} {'Fisher':>10s} {'MCMC':>10s}  ratio")
    mcmc = [c[:, 0].std(), c[:, 1].std(), c[:, 2].std()]
    for i, n in enumerate(NAMES):
        print(f"  sig({n:2s})  {arr[k, 2+i]:10.3e} {mcmc[i]:10.3e}  {arr[k,2+i]/mcmc[i]:5.2f}")
    print(f"  rho      {arr[k,5]:+10.3f} {np.corrcoef(c[:,1],c[:,2])[0,1]:+10.3f}")
    print("\n  Fisher is prior-free; the MCMC row carries a tight box, so expect")
    print("  Fisher >= MCMC for m1/m2. A ratio near 1 at e=0.3 confirms that")
    print("  chain was genuinely likelihood-dominated.")
