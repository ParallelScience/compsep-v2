# compsep-v2

**Scientist:** denario-6
**Date:** 2026-04-15

# Dataset Description: FLAMINGO Lensed Component-Separation Maps

This dataset consists of flat-sky cut maps derived from full-sky maps of the cosmic microwave background (CMB), thermal and kinetic Sunyaev-Zel'dovich effects (tSZ/kSZ), and the cosmic infrared background (CIB). The sky signal maps originate from the FLAMINGO L1_m9 HYDRO_FIDUCIAL simulation (lightcone0, Jeger rotation) — a large-volume (1 Gpc comoving box) hydrodynamical simulation with self-consistent baryonic physics (radiative cooling, star formation, AGN feedback, diffuse gas flows).

The CMB is a Gaussian realization lensed by the FLAMINGO convergence map (κ) via lenspyx, using Planck-like cosmology (h=0.6736, n_s=0.965, A_s=2.1e-9, seed=42).

## File Inventory (absolute paths, verified)

All files are under `/home/node/data/compsep_data/cut_maps/`.

### Individual component maps — ground truth, 1 arcmin beam
Shape: **(1523, 256, 256)**, dtype **float64**

- `/home/node/data/compsep_data/cut_maps/lensed_cmb.npy` — Lensed CMB temperature, µK_CMB
- `/home/node/data/compsep_data/cut_maps/tsz.npy` — tSZ Compton-y parameter, dimensionless
- `/home/node/data/compsep_data/cut_maps/ksz.npy` — kSZ Doppler-b parameter, dimensionless
- `/home/node/data/compsep_data/cut_maps/cib_90.npy` — CIB at 90 GHz, Jy/sr (delta-bandpass)
- `/home/node/data/compsep_data/cut_maps/cib_150.npy` — CIB at 150 GHz, Jy/sr (delta-bandpass)
- `/home/node/data/compsep_data/cut_maps/cib_217.npy` — CIB at 217 GHz, Jy/sr (bandpass-integrated)
- `/home/node/data/compsep_data/cut_maps/cib_353.npy` — CIB at 353 GHz, Jy/sr (bandpass-integrated)
- `/home/node/data/compsep_data/cut_maps/cib_545.npy` — CIB at 545 GHz, Jy/sr (bandpass-integrated)
- `/home/node/data/compsep_data/cut_maps/cib_857.npy` — CIB at 857 GHz, Jy/sr (bandpass-integrated)

### Stacked (observed) signal maps — frequency-specific beam, no noise
Shape: **(1523, 256, 256)**, dtype **float64**, units **µK_CMB**

- `/home/node/data/compsep_data/cut_maps/stacked_90.npy` — 90 GHz total signal, beam FWHM 2.2 arcmin (SO LAT)
- `/home/node/data/compsep_data/cut_maps/stacked_150.npy` — 150 GHz total signal, beam FWHM 1.4 arcmin (SO LAT)
- `/home/node/data/compsep_data/cut_maps/stacked_217.npy` — 217 GHz total signal, beam FWHM 1.0 arcmin (SO LAT)
- `/home/node/data/compsep_data/cut_maps/stacked_353.npy` — 353 GHz total signal, beam FWHM 4.5 arcmin (Planck HFI)
- `/home/node/data/compsep_data/cut_maps/stacked_545.npy` — 545 GHz total signal, beam FWHM 4.72 arcmin (Planck HFI)
- `/home/node/data/compsep_data/cut_maps/stacked_857.npy` — 857 GHz total signal, beam FWHM 4.42 arcmin (Planck HFI)

The stacked signal is the linear mixture of all components in µK_CMB:
  signal(freq) = CIB(freq)*jysr2uk(freq) + tSZ*f_tSZ(freq) + kSZ*f_kSZ + lensed_CMB
then smoothed with the frequency-specific beam. Individual component maps have a 1 arcmin beam.

### Noise maps

**Simons Observatory (SO) LAT noise** — 3000 independent Gaussian realizations
Shape: **(3000, 256, 256)**, dtype **float64**, units **µK_CMB**

- `/home/node/data/compsep_data/cut_maps/so_noise/90.npy`
- `/home/node/data/compsep_data/cut_maps/so_noise/150.npy`
- `/home/node/data/compsep_data/cut_maps/so_noise/217.npy`

SO noise is drawn from the SO LAT v3.1 temperature noise power spectrum (mode 2, elevation 50°, f_sky=0.4). 90 and 150 GHz are correlated (Cholesky decomposition); 217 GHz is independent. When sampling SO noise, the noise index does NOT need to match the sky patch index.

**Planck FFP10 noise** — 100 MC realizations per high-frequency channel
Shape: **(1523, 256, 256)**, dtype **float64**
Filename pattern: `planck_noise_{freq}_{i}.npy`, freq ∈ {353, 545, 857}, i ∈ {0, …, 99}

- `/home/node/data/compsep_data/cut_maps/planck_noise/planck_noise_353_{0..99}.npy` — units: K_CMB → multiply by 1e6 to get µK_CMB
- `/home/node/data/compsep_data/cut_maps/planck_noise/planck_noise_545_{0..99}.npy` — units: MJy/sr → multiply by 1e6 * jysr2uk(545) to get µK_CMB
- `/home/node/data/compsep_data/cut_maps/planck_noise/planck_noise_857_{0..99}.npy` — units: MJy/sr → multiply by 1e6 * jysr2uk(857) to get µK_CMB

For Planck noise, the patch index MUST match the sky patch index.

## Spectral Response and Utility Functions

A `utils.py` module is available at `/home/node/data/compsep_data/utils.py`. **Always import it using the absolute path or by adding its directory to sys.path:**

```python
import sys
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
```

Key functions in utils.py:
- `utils.tsz(freq_ghz)` — tSZ spectral response in µK_CMB per Compton-y unit
- `utils.ksz(freq_ghz)` — kSZ spectral response in µK_CMB (= -T_CMB_µK, frequency-independent)
- `utils.jysr2uk(freq_ghz)` — converts Jy/sr to µK_CMB for CIB at given frequency
- `utils.powers(a, b, ps=10, ell_n=199, window_alpha=None)` — flat-sky angular auto/cross power spectrum
- `utils.get_patch_centers(gal_cut, step_size)` — returns list of (lon, lat) patch centers

## Patch Geometry

- 1523 patches, 5°×5°, 256×256 pixels, ≈1.17 arcmin/pixel
- Full-sky tessellation at 5° step spacing (no galactic cut)
- Gnomonic (flat-sky) projection, bilinear interpolation from HEALPix nside=8192

## Loading Data (correct absolute paths)

```python
import sys
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
import numpy as np

BASE = '/home/node/data/compsep_data/cut_maps'

rng = np.random.default_rng(seed=42)
n_patch = 1523
n_planck = 100
n_so = 3000

i_patch = rng.integers(n_patch)   # sky patch index
i_planck = rng.integers(n_planck) # Planck MC realization index
i_so = rng.integers(n_so)         # SO noise index (does not need to match i_patch)

# Load observed signal + noise at each frequency
frequencies = [90, 150, 217, 353, 545, 857]
patches = {}
for freq in frequencies:
    signal = np.load(f'{BASE}/stacked_{freq}.npy')[i_patch]  # µK_CMB
    if freq <= 217:  # SO noise
        noise = np.load(f'{BASE}/so_noise/{freq}.npy')[i_so]  # µK_CMB
    else:  # Planck noise
        raw = np.load(f'{BASE}/planck_noise/planck_noise_{freq}_{i_planck}.npy')[i_patch]
        if freq == 353:
            noise = raw * 1e6                        # K_CMB -> µK_CMB
        else:
            noise = raw * 1e6 * utils.jysr2uk(freq) # MJy/sr -> µK_CMB
    patches[freq] = signal + noise  # µK_CMB

# Load ground truth components
cmb   = np.load(f'{BASE}/lensed_cmb.npy')[i_patch]  # µK_CMB
tsz   = np.load(f'{BASE}/tsz.npy')[i_patch]          # Compton-y (dimensionless)
ksz   = np.load(f'{BASE}/ksz.npy')[i_patch]          # Doppler-b (dimensionless)
cib90 = np.load(f'{BASE}/cib_90.npy')[i_patch]       # Jy/sr
```

## CMB Simulation Cosmology

CLASS parameters used:
- A_s = 2.1e-9, n_s = 0.965, h = 0.6736
- ω_b = 0.02237, ω_cdm = 0.12, τ_reio = 0.0544, Y_He = 0.2454

## Hardware and Environment

- Python: `/opt/denario-venv/bin/python`
- *64 vCPUs* (AMD Ryzen Threadripper PRO 9995WX), *128 GB RAM*
- *NVIDIA RTX PRO 6000 Blackwell Edition, 96 GB VRAM*, CUDA 13.0
- PyTorch GPU: use `device='cuda'`
- Multiprocessing: limit to ~8–16 workers to avoid oversubscription
- Set `OMP_NUM_THREADS` when using NumPy/SciPy with multiprocessing
- Load data patch-by-patch or in small batches; avoid loading all 1523 patches for all frequencies simultaneously (total data volume is large)
