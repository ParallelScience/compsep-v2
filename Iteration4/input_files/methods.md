1. **Data Pipeline and Noise Conditioning**:
    - Implement a `Dataset` class that dynamically samples SO and Planck noise realizations.
    - Provide the model with the local noise variance (per-patch, per-frequency) as an auxiliary conditioning vector to allow the model to adapt its internal thresholding logic.
    - Implement robust input normalization: use log-transforms or median/IQR scaling for CIB channels to manage the high dynamic range and prevent gradient dominance.
    - Stratify the 1523-patch split by halo mass/tSZ intensity to ensure the validation set contains a representative sample of high-mass clusters for $Y_{SZ}-M$ analysis.

2. **Architecture and Spectral Fidelity**:
    - Utilize the multi-scale U-Net with gated cross-attention.
    - Implement a composite loss function: pixel-wise L1 reconstruction loss plus a spectral consistency term.
    - Apply the spectral loss to the *log* of the power spectrum, using $\ell^2$ or $\ell^3$ weighting to ensure high-$\ell$ modes (baryonic feedback) are not overwhelmed by low-$\ell$ power.
    - Use a dynamic weighting schedule for the spectral loss ($\lambda_3$), increasing its influence as training progresses to prioritize coarse structure before fine-scale spectral constraints.

3. **Transfer Function Analysis**:
    - Compute the transfer function $T(\ell) = \sqrt{P_{recon}(\ell) / P_{truth}(\ell)}$ using `utils.powers` with Hann windowing to mitigate edge effects.
    - Monitor $T(\ell)$ during training; if high-frequency recovery ($\ell > 5000$) lags, adjust $\lambda_3$ to prevent the model from acting as a low-pass filter.

4. **Causal Validation and Hallucination Testing**:
    - Perform "CIB-only" reconstruction: run the model with CIB inputs while setting SO-LAT channels to pure noise.
    - Quantify hallucination risk by calculating the pixel-wise correlation between the CIB-only output and the ground truth tSZ map. A non-zero correlation indicates the model is mapping CIB features to tSZ rather than reconstructing gas pressure.

5. **Residual Analysis for Bias Detection**:
    - Calculate residuals $R = \text{Truth} - \text{Reconstruction}$.
    - Perform correlation analysis between $R$ and the input/truth in both the pixel domain and the Fourier domain. This identifies if bias is scale-dependent (e.g., accurate at large scales but biased at small scales).

6. **Curriculum Learning and Stability**:
    - Execute two-stage training: Stage 1 (high-mass clusters) followed by Stage 2 (full dataset).
    - Define a "High-Mass Cluster Benchmark" (fixed subset) to evaluate at the end of both stages.
    - Quantify "forgetting" by measuring the change in $Y_{SZ}-M$ scatter for the top 10% most massive clusters to ensure stability.

7. **Scientific Validation: $Y_{SZ}-M$ Relation**:
    - Cross-match reconstructed maps with the FLAMINGO halo catalog to extract the $Y_{SZ}-M$ relation.
    - Compare the scatter of the reconstructed $Y_{SZ}-M$ relation against the ground truth to determine if the model preserves physical variance or regresses to the mean.

8. **Final Performance Reporting**:
    - Consolidate metrics: global cross-correlation coefficient $r_\ell$ (using tapered windows), transfer function $T(\ell)$, residual correlation coefficients (pixel/Fourier), and $Y_{SZ}-M$ scatter.
    - Provide a comprehensive assessment of the model’s physical fidelity and robustness to noise.