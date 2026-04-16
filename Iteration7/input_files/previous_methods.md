1. **Data Preprocessing and Pipeline Optimization**:
    - Implement a multi-threaded data loader using `multiprocessing` to pre-fetch and normalize patches in the background.
    - Pre-calculate and cache `jysr2uk` conversions and unit scaling factors to minimize redundant computations.
    - Apply standard scaling based on global training set statistics and implement random rotations/flips for rotational invariance.
    - Cache the entire validation set (or a representative subset) in RAM to ensure high-multipole metrics are computed over the full 1523 patches.

2. **SR-DAE Architecture and Training**:
    - Construct a multi-scale U-Net with gated cross-attention layers, using SO-LAT channels (90, 150, 217 GHz) as primary inputs and Planck channels (353–857 GHz) as auxiliary features.
    - Train using the composite loss $\mathcal{L} = \mathcal{L}_{L1} + \lambda_1 \mathcal{L}_{spec} + \lambda_2 \mathcal{L}_{corr}$.
    - Define $\mathcal{L}_{spec}$ using `utils.powers` for $\ell \in [1000, 5000]$, ensuring the training objective is aligned with the validation metrics.

3. **Quantifying CIB-to-tSZ Leakage**:
    - Compute the residual map $R = \text{tSZ}_{true} - \text{tSZ}_{pred}$.
    - Calculate the cross-correlation coefficient between $R$ and input CIB maps at each frequency to detect signal leakage.
    - Use these coefficients as a primary metric to ensure the model avoids hallucinating tSZ features to compensate for CIB foregrounds.

4. **Scientific Validation via $Y_{SZ}-M$ Relation**:
    - Extract halo masses ($M_{500c}$) from the FLAMINGO catalog.
    - Calculate $Y_{SZ}$ by integrating the reconstructed map over the specific $R_{500c}$ aperture centered on the halo to ensure physical consistency.
    - Perform regression of $Y_{SZ}$ against $M_{500c}$ and compare the scatter and bias against the linear cILC baseline.

5. **Transfer Function and Resolution Analysis**:
    - Compute the transfer function $T(\ell) = \frac{P_{cross}(\text{pred}, \text{true})}{P_{auto}(\text{true})}$.
    - Explicitly compare the SR-DAE transfer function against the cILC transfer function to quantify the resolution gain beyond the input beam limits (2.2–4.72 arcmin).
    - Include 1-sigma error bars derived from patch-to-patch variance to validate performance at $\ell \approx 5000$.

6. **Baseline Comparison**:
    - Implement a constrained Internal Linear Combination (cILC) as the primary linear benchmark, configured to null CIB spectral signatures while preserving tSZ.
    - Compare SR-DAE against cILC using MSE, bias, and $Y_{SZ}-M$ scatter metrics.

7. **Conditional Diffusion Model (CDM) Refinement**:
    - Only implement the CDM if the SR-DAE bias in the $Y_{SZ}-M$ relation exceeds the cILC baseline bias.
    - If implemented, use the SR-DAE as the backbone to learn $p(\text{tSZ} | \text{Observed}, \text{Noise})$.
    - Adopt the CDM only if the variance-weighted $Y_{SZ}-M$ regression yields lower scatter or bias than the standard SR-DAE; otherwise, omit the CDM from the final report.

8. **Final Performance Reporting**:
    - Consolidate all metrics: $T(\ell)$, $Y_{SZ}-M$ scatter, residual-CIB correlation, and bias measurements.
    - Document the performance advantage over linear methods and provide evidence of physical validity through the $Y_{SZ}-M$ relation.