1. **Data Pipeline and Input Integration**:
    - Construct input tensors by concatenating the six frequency maps (90, 150, 217, 353, 545, 857 GHz).
    - Implement a high-performance data loader using `torch.utils.data.DataLoader` with `pin_memory=True` and `num_workers=16`.
    - Dynamically sample SO noise (3000 realizations) and Planck noise (100 realizations) per iteration. Apply random noise augmentation (scaling by 0.9–1.1) to enhance robustness.
    - Apply random 90-degree rotations and flips to all patches to improve rotational invariance. Normalize inputs using global training set statistics.

2. **Architecture Design**:
    - Implement a multi-scale U-Net with gated cross-attention modules at each decoder level.
    - Use the low-frequency channels (90, 150, 217 GHz) as the primary spatial stream and high-frequency CIB channels (353–857 GHz) as the auxiliary stream.
    - The network output is a single-channel map representing the reconstructed tSZ Compton-y parameter.

3. **Loss Function Formulation**:
    - Define a composite loss: $L = \lambda_1 L_{L1} + \lambda_2 L_{FeatureMatching} + \lambda_3 L_{Spectral}$.
    - $L_{Spectral}$ is computed using `utils.powers` by comparing the batch-averaged radial power spectrum of the reconstructed maps against the ground truth maps, ensuring consistent application of the `window_alpha` function to both.
    - Use a continuous weighting function based on ground truth intensity to prioritize high-mass clusters while maintaining sensitivity to diffuse gas.

4. **Curriculum Learning Strategy**:
    - Stage 1: Train on high-intensity patches (high-mass clusters) to establish the fundamental mapping between frequency signals and gas pressure.
    - Stage 2: Introduce the full dataset, including low-mass, low-SNR patches, to allow the model to learn subtle, diffuse gas signatures.

5. **CIB-tSZ Null Test (Causal Validation)**:
    - Create a control dataset by shuffling CIB patch indices relative to fixed tSZ ground truth maps, ensuring the CIB statistical properties (mean, variance, power spectrum) remain identical to the original set.
    - Maintain the original noise realizations in the control set to isolate the effect of spatial correlation. If the model fails to reconstruct the tSZ signal, it confirms the model learns physical gas pressure rather than CIB-morphology painting.

6. **Training Configuration**:
    - Use the AdamW optimizer with a one-cycle learning rate scheduler.
    - Perform training in mixed-precision (FP16) on the NVIDIA RTX PRO 6000.
    - Set `OMP_NUM_THREADS=8` to optimize CPU-side data loading.

7. **Scientific Validation: Pressure-Mass Relation**:
    - Extract the $Y_{SZ}-M$ relation by cross-matching reconstructed maps with known halo masses.
    - Calculate the scatter (standard deviation of residuals) as a function of mass to detect "collapsing" of physical variance.
    - Compare the scatter and the $Y_{SZ}-M$ relation against both the ground truth and the "raw" (unprocessed) SO-LAT data to quantify information gain.

8. **Performance Assessment**:
    - Quantify reconstruction quality using the cross-correlation coefficient $r_\ell$ globally and within mass bins.
    - Perform a "Bias-Variance" analysis on residuals as a function of tSZ intensity to ensure the model preserves the physical scatter of baryonic feedback.