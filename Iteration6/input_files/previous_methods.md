1. **Baseline Implementation and Comparative Benchmarking**:
    - Implement a constrained Internal Linear Combination (cILC) and a Wiener Filter as primary linear baselines. Configure the cILC to null the CIB spectral signature while preserving the tSZ signal, using weights derived from the FLAMINGO component spectra.
    - Optimize baseline weights to minimize reconstruction variance while maintaining a unit response to the tSZ spectral signature.
    - Use `multiprocessing` to pre-load and augment patches in parallel, ensuring the GPU remains saturated.

2. **Data Pipeline and Out-of-Distribution (OOD) Stress Testing**:
    - Implement a "Leave-One-Out" strategy by isolating the top 5% most massive clusters from the training set to test generalization to extreme feedback morphologies.
    - Perform a "noise-level" stress test by evaluating performance on the 95th percentile of SO noise realizations to ensure the model does not hallucinate features in low-SNR regimes.
    - Conduct cross-patch validation on subsets with varying CIB-to-tSZ ratios to assess robustness against foreground contamination levels.

3. **Architecture and Feature Attribution**:
    - Utilize a multi-scale U-Net with gated cross-attention.
    - Implement Integrated Gradients (IG) to generate saliency maps, verifying that the model actively utilizes SO-LAT channels for pressure reconstruction rather than relying solely on CIB priors.
    - Perform a "Null Test" by running the model on pure noise realizations to ensure the output is statistically consistent with zero.

4. **Spectral Fidelity and Loss Function Optimization**:
    - Define a composite loss function: $\mathcal{L} = \mathcal{L}_{L1} + \lambda_1 \mathcal{L}_{spec} + \lambda_2 \mathcal{L}_{corr}$.
    - $\mathcal{L}_{spec}$ uses `utils.powers` on the log of the power spectrum with $\ell^3$ weighting to emphasize high-frequency baryonic feedback.
    - $\mathcal{L}_{corr}$ is a differentiable normalized cross-correlation penalty between the reconstruction residual and the input CIB maps, forcing the model to learn a representation orthogonal to CIB foregrounds.

5. **Probabilistic Framework Transition (Stage 2)**:
    - Transition the SR-DAE to a Conditional Diffusion Model (CDM) by using the pre-trained SR-DAE weights as the backbone.
    - Train the CDM to learn the conditional distribution $p(\text{tSZ} | \text{Observed}, \text{Noise})$ using pre-calculated noise schedules and mixed-precision training.
    - Use the diffusion process to generate multiple realizations, enabling the calculation of pixel-wise variance maps as a proxy for model uncertainty.

6. **Uncertainty Calibration and Validation**:
    - Validate the calibration of the CDM uncertainty by checking if pixel-wise variance maps are statistically consistent with actual reconstruction errors (e.g., via Probability Integral Transform).
    - Verify that predicted $\sigma$ intervals correctly capture increased uncertainty in regions with high SO noise or high CIB contamination.

7. **Comparative Scientific Validation**:
    - Extract the $Y_{SZ}-M$ relation from the reconstructed maps using the FLAMINGO halo catalog.
    - Compare the scatter and bias of the SR-DAE/CDM against the optimized linear baselines.
    - Quantify the "gain" of the model by calculating the ratio of reconstruction error (MSE) and $Y_{SZ}-M$ scatter relative to the linear baselines across $\ell \in [1000, 5000]$.

8. **Final Performance Reporting**:
    - Consolidate metrics: $T(\ell)$, $Y_{SZ}-M$ scatter, residual-CIB correlation coefficients, and uncertainty calibration scores.
    - Summarize the performance advantage over linear methods, robustness to OOD merger events, and the physical validity of the reconstructed pressure fluctuations.