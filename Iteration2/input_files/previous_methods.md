1. **Evaluation Pipeline and Coordinate Validation**:
    - Verify the `utils.powers` estimator by computing the power spectrum of a known Gaussian random field and comparing it against the theoretical input spectrum to ensure correct pixel-to-physical mapping (1.17 arcmin/pixel).
    - Perform a coordinate alignment check: compute the cross-correlation of the ground truth tSZ map with a 90-degree rotated version of itself to ensure zero correlation, confirming no systematic spatial offsets or phase-incoherence.
    - Implement a diagnostic check to compare the mean and variance of the ground truth vs. model output to identify potential mean-subtraction errors.

2. **Data Pipeline and Preprocessing**:
    - Implement a `torch.utils.data.Dataset` that enforces index matching for Planck noise (`i_patch == i_planck`) while sampling SO noise independently (`i_so` random) to prevent the model from memorizing specific noise patterns.
    - Apply channel-wise robust scaling (median and interquartile range) to all input maps, particularly CIB channels, to prevent outliers from dominating the input features.
    - Apply a soft-clipping transformation to the tSZ ground truth to manage high dynamic range while preserving filamentary signal.

3. **Refined Architecture with Interpretability**:
    - Utilize the multi-scale U-Net with gated cross-attention, injecting CIB features at the bottleneck and skip connections.
    - Integrate an Integrated Gradients module to generate saliency maps, using a "neutral" baseline (zeros or mean background). Perform this analysis across both high-mass clusters and low-mass filaments to determine if the CIB-attention acts as a cluster-finder or a diffuse gas reconstructor.
    - Normalize saliency maps by input variance to ensure gradients reflect informative features rather than numerical magnitude.

4. **Differentiable Multi-Resolution Wavelet Loss**:
    - Implement a differentiable wavelet decomposition (using a custom filter bank or `pytorch-wavelets`) to penalize deviations specifically in the 1–5 arcmin scale range.
    - Implement a Differentiable Pseudo-$C_\ell$ estimator that incorporates the apodization window to compute the power spectrum loss for $\ell > 1000$.

5. **Physical Calibration and Flux Constraints**:
    - Define a composite loss: $L = \lambda_1 L_{L1} + \lambda_2 L_{wavelet} + \lambda_3 L_{pseudo-Cl} + \lambda_4 L_{flux}$.
    - Implement $L_{flux}$ as a combination of global sum conservation and a "local" flux constraint using a Gaussian-weighted sum to preserve pressure in the vicinity of individual clusters.
    - Anneal the weight $\lambda_4$ during training, starting low to allow structural convergence before enforcing strict flux conservation.

6. **Signal-Injection Testing**:
    - Replace pure noise null tests with a Signal-Injection Test: inject a known GNFW cluster profile into pure noise maps.
    - Measure the recovery accuracy of the integrated $Y$ and the radial profile of the injected cluster to quantify the model's ability to reconstruct physical structures in noise-dominated regimes.

7. **Training and Optimization**:
    - Use the AdamW optimizer with a one-cycle learning rate scheduler.
    - Select a validation set of 150 patches that spans a representative range of FLAMINGO density environments (clusters vs. filaments).
    - Use mixed-precision training (FP16) on the RTX PRO 6000, ensuring `OMP_NUM_THREADS` is set to avoid oversubscription.

8. **Performance Assessment**:
    - Evaluate the model using the cross-correlation coefficient $r_\ell$ as a function of multipole $\ell$ to ensure phase coherence.
    - Perform a "Bias-Variance" check: calculate the mean of the residuals (pred - truth) as a function of the ground truth tSZ intensity to identify systematic underestimation in high-density regions or overestimation in low-density regions.
    - Quantify reconstruction quality via the recovery of integrated $Y$ across different cluster mass bins.