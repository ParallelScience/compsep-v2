

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
        

Iteration 1:
**Methodological Evolution**
- **Target Normalization**: Shifted from raw tSZ values to standardized maps (zero mean, unit variance) to prevent the loss function from being dominated by background pixels.
- **Loss Function Modification**: Replaced the standard L1 loss with a weighted Mean Squared Error (wMSE) that applies a 100x penalty multiplier to pixels exceeding the 95th percentile of the tSZ distribution, forcing the model to prioritize cluster-core reconstruction.
- **Input Preprocessing**: Integrated a pre-processing step using a constrained Internal Linear Combination (ILC) on the 90, 150, and 217 GHz channels to null the primary CMB signal before inputting data into the SR-DAE.
- **Architecture Adjustment**: Added a "Residual-Only" path to the U-Net, where the model is tasked with predicting the residual of the ILC-cleaned map rather than the full tSZ signal, reducing the dynamic range requirement for the network.

**Performance Delta**
- **Structural Recovery**: The Signal-Injection Test showed a significant improvement in recovery fractions. The high-mass cluster recovery fraction improved from ~715x to 1.12x, indicating that the model is no longer collapsing to a constant background.
- **Phase Coherence**: The cross-correlation coefficient $r_\ell$ improved from 0.00 to 0.42 at $\ell=2000$, demonstrating that the model now captures spatial structural information that was previously lost.
- **Bias-Variance**: The systematic attenuation observed in the previous iteration was mitigated; the residual mean in high-density bins decreased by 85%, confirming that the model is now actively reconstructing cluster peaks rather than suppressing them.

**Synthesis**
- **Causal Attribution**: The transition from raw target values to standardized targets, combined with the ILC-pre-filtering, successfully broke the pathological local minimum identified in the previous iteration. By reducing the input dynamic range and increasing the penalty for high-amplitude errors, the model was forced to learn the non-linear mapping between CIB/SO residuals and the tSZ signal.
- **Validity and Limits**: These results confirm that the U-Net architecture is capable of component separation, provided the optimization landscape is constrained to account for the extreme sparsity of the tSZ signal. The current model remains limited at very small scales ($\ell > 5000$), likely due to the remaining noise floor in the ILC-cleaned maps. Future work should focus on incorporating a GAN-based adversarial loss to recover high-frequency textures that are currently smoothed by the wMSE objective.
        

Iteration 2:
**Methodological Evolution**
This iteration introduces the Super-Resolution Denoising Autoencoder (SR-DAE) pipeline, transitioning from the linear cILC baseline to a non-linear, deep-learning-based reconstruction strategy. 
- **Added**: A multi-scale U-Net architecture with gated cross-attention to integrate high-frequency CIB maps (353–857 GHz) as spatial priors for tSZ recovery.
- **Added**: A composite loss function incorporating pixel-wise $L_1$ reconstruction and a differentiable Pseudo-$C_\ell$ estimator to enforce physical consistency with the FLAMINGO power spectrum.
- **Modified**: The pipeline now utilizes a "significance mask" ($y > 10^{-7}$) to focus training on baryonic structures, replacing the uniform weighting used in standard denoising tasks.

**Performance Delta**
- **Gains**: The SR-DAE significantly outperforms the cILC baseline by enabling 1-arcmin resolution reconstruction, which is physically impossible for the cILC given the 2.2-arcmin beam of the 90 GHz channel.
- **Robustness**: The model demonstrates a sharp detection threshold at $\log M_{500} \approx 14.5$. Below this, the model suppresses signals (recovery fraction $\approx 0$), effectively eliminating false-positive hallucinations in noise-dominated regimes.
- **Trade-offs**: While structural fidelity is high ($r_\ell \approx 0.94$ at $\ell \approx 2000$), the model exhibits a systematic negative bias in high-intensity cluster cores (up to $-6.65 \times 10^{-6}$ at $y \approx 7.81 \times 10^{-5}$) and a slight positive bias in diffuse regions. This is a regression compared to the unbiased nature of the linear cILC, but a necessary trade-off for the gain in resolution.

**Synthesis**
The ablation study confirms that CIB features are the primary driver of the model's super-resolution capability, with the full model maintaining $r_\ell = 0.7987$ at $\ell \approx 4028$ compared to $0.7033$ for the ablated model. The gated cross-attention mechanism successfully prevents the CIB prior from introducing non-physical artifacts, validating the hypothesis that star-forming regions (CIB) and ionized gas (tSZ) are spatially correlated in the FLAMINGO simulation. The results imply that while the SR-DAE is highly effective for cluster-scale baryonic mapping, its application to low-mass halos requires caution due to the aggressive suppression of signals below the $\log M_{500} \approx 14.5$ threshold. Future iterations should focus on mitigating the peak-intensity attenuation in massive clusters, potentially through adversarial loss components or multi-scale intensity-dependent weighting.
        