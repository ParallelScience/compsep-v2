1. **Data Pre-processing and cILC Construction**:
    - Calculate cILC weights using the empirical noise covariance matrices derived from the provided SO noise realizations to null the primary CMB signal.
    - Use cross-patch covariance estimates to ensure weight stability and prevent overfitting to specific noise realizations.
    - Apply a regularization term to the cILC weights to minimize residual variance in regions where tSZ signal is negligible, mitigating CIB/kSZ leakage.

2. **Feature Engineering and Normalization**:
    - Construct spectral difference maps (e.g., $150-90$ GHz, $217-150$ GHz) to highlight the tSZ spectral signature.
    - Normalize both the spectral difference maps and the CIB auxiliary channels (353–857 GHz) by their respective local or global noise variance to ensure consistent dynamic ranges.
    - Apply log-scaling or robust scaling to CIB maps to prevent high-intensity foregrounds from dominating the input tensor.

3. **Target Normalization and Masking**:
    - Standardize tSZ ground truth maps to unit variance.
    - Generate a binary "significance mask" based on a threshold of the ground truth tSZ ($y > 10^{-7}$) to identify physically significant baryonic structures.
    - Implement geometric data augmentation (random 90-degree rotations and flips) to ensure rotational invariance and improve model generalization.

4. **Focal Loss and Training Stability**:
    - Implement a Focal Loss function with a weighting factor of $10^3–10^4$ for pixels within the significance mask to prioritize high-intensity baryonic features.
    - Incorporate gradient clipping to prevent potential gradient explosions caused by the high dynamic range of the Focal Loss.
    - Use the AdamW optimizer with a one-cycle learning rate scheduler and mixed-precision (FP16) training.

5. **Architecture and Training Configuration**:
    - Deploy a multi-scale U-Net with gated cross-attention to process the concatenated spectral difference and CIB auxiliary features.
    - Set `OMP_NUM_THREADS` to avoid oversubscription and utilize the RTX PRO 6000 for accelerated training.
    - Monitor training using both "significant" masked regions and "null" patches (empty regions) to detect and prevent hallucination artifacts.

6. **Physical Consistency and Power Spectrum Constraints**:
    - Integrate a differentiable Pseudo-$C_\ell$ estimator into the loss function to penalize deviations from the FLAMINGO-simulated tSZ power spectrum at $\ell > 1000$.
    - Apply a wavelet-based loss to ensure structural fidelity specifically in the 1–5 arcmin range, where baryonic feedback signatures are most prominent.

7. **Sensitivity and Ablation Analysis**:
    - Perform a sensitivity analysis by training a control model without CIB features to quantify the performance gain versus potential bias introduced by CIB contamination.
    - Inject GNFW cluster profiles of varying masses and peak-y values into noise-dominated inputs to determine the minimum mass/peak-y threshold for reliable reconstruction.

8. **Performance Assessment**:
    - Evaluate the model using the cross-correlation coefficient $r_\ell$ computed both globally and specifically within the masked regions.
    - Perform a "Bias-Variance" check on residuals as a function of tSZ intensity to identify systematic underestimation in high-density regions.
    - Quantify reconstruction quality across different cluster mass bins to validate the model's utility for baryonic pressure mapping.