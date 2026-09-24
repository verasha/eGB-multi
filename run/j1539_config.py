"""Shared prior configuration for the ZTF J1539+5027 eccentricity scan.

Single source of truth for the MASS prior box, so std(m1)/std(m2)/std(M) are
directly comparable across every run in the scan. Sized from the SMALLEST
non-zero eccentricity in the scan (e=0.1), where the masses are worst
constrained: Fisher gives sigma(m1)=2.03e-2, sigma(m2)=5.87e-3 there, and the
box is 6 sigma on each side so the posterior stays interior.

Do not retune this per run. A per-run box makes the scan incomparable -- the
mistake that made the HM Cnc e=0/e=0.1 rows unusable.

The ECCENTRICITY prior legitimately differs per run (it brackets each
injection), so it stays in the individual scripts.

Caveat for e=0: the mass block is numerically singular there, so NO finite box
contains the posterior and std(m1) will simply be the box width / sqrt(12).
Report that row as "unconstrained", never as a sigma. Only sigma(Mc) is a real
measurement at e=0 -- that direction is interior by ~22x.
"""

import numpy as np

# truth: m1 = 0.61, m2 = 0.21  (Burdge et al. 2019)
MASS_BOUNDS = np.array([[0.49, 0.73],    # m1: 0.61 +/- 6*2.03e-2
                        [0.17, 0.25]])   # m2: 0.21 +/- 6*5.87e-3

# Proposal step for the masses, set by the NARROW direction across the
# degeneracy ridge: sigma(Mc) ~ 6.7e-4 at both e=0 and e=0.1.
MASS_SIGMA = 0.002
