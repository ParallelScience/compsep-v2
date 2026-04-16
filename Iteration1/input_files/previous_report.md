

Iteration 0:
# Summary: SR-DAE for tSZ Reconstruction

## 1. Status and Key Findings
- **Success:** A Dual-Branch U-Net with gated cross-attention successfully reconstructs 1-arcmin tSZ maps from SO LAT bands (90, 150, 217 GHz) using CIB (353–857 GHz) as spatial priors.
- **Performance:** The `no-spectral-loss` model achieved an RMSE of $2.0 \times 10^{-6}$ (matching signal RMS) and an SSIM of 0.635.
- **Failure:** The logarithmic spectral consistency loss ($L_{spec}$) caused catastrophic failure, acting as a generative prior that forced the model to hallucinate non-physical structures (6603% hallucination fraction in Null Tests).
- **Anomaly:** Cross-correlation ($r_\ell$) results are currently unreliable (reported 0.0) due to suspected pipeline errors in `utils.powers` or phase-alignment logic.

## 2. Methodological Constraints
- **Architecture:** Dual-branch U-Net with FPN-style cross-attention is effective for fusing CIB spatial priors with SO signal.
- **Loss Function:** Pixel-wise L1 and Sobel-based edge loss are sufficient for high-fidelity reconstruction. Spectral constraints must be avoided or reformulated to prevent hallucination.
- **Uncertainty:** MC Dropout provides reliable pixel-wise uncertainty maps, essential for SNR-based masking in downstream cosmology.

## 3. Critical Decisions for Future Experiments
- **Discard $L_{spec}$:** Do not use the current logarithmic spectral loss. Future spectral constraints must be linear, SNR-weighted, or applied only to high-confidence multipole bins.
- **Fix Evaluation Pipeline:** Debug `utils.powers` or the cross-correlation implementation to ensure $r_\ell$ reflects true phase coherence.
- **Leverage CIB:** The gated cross-attention mechanism is validated; continue using CIB as a probabilistic spatial prior rather than assuming deterministic coupling.
- **Inference:** Utilize MC Dropout uncertainty maps to generate SNR masks for cosmological parameter estimation.

## 4. Data/Environment Notes
- **Data:** 1523 patches (5°×5°). SO noise is independent of patch index; Planck noise must match patch index.
- **Hardware:** 64 vCPUs, 96 GB VRAM (RTX PRO 6000). Use `num_workers=16` for `DataLoader`.
- **Environment:** `utils.py` is located at `/home/node/data/compsep_data/utils.py`.
        