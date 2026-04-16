1. **Data Preprocessing and Memory Management**:
   - Pre-load all 1523 ground truth maps (tSZ, kSZ, CMB, CIB) into system RAM at initialization to maximize I/O efficiency.
   - Apply a non-linear transformation (e.g., arcsinh) to tSZ and CIB maps to compress the dynamic range and enhance the visibility of both high-intensity clusters and low-intensity filaments.
   - Implement a `torch.utils.data.DataLoader` with `num_workers=16` to handle on-the-fly noise injection. For each training iteration, sample SO noise (indices 0–2999) and Planck noise (indices 0–99) and add them to the stacked signal maps.
   - Pass the noise power spectrum or local noise variance maps as auxiliary input channels to the U-Net to allow the model to dynamically adjust its denoising strength.

2. **Architecture Design (Multi-scale U-Net with FPN-style Cross-Attention)**:
   - Construct a U-Net where the encoder branches process SO-band maps (90, 150, 217 GHz) and high-frequency CIB-band maps (353–857 GHz) through separate feature extractors.
   - Implement a Feature Pyramid Network (FPN) style architecture to inject CIB spectral features into the U-Net decoder at corresponding spatial scales, ensuring high-frequency CIB information guides the recovery of small-scale gas pressure fluctuations.
   - Integrate gated cross-attention modules at each resolution level to modulate the tSZ reconstruction based on the CIB-traced star-forming regions.

3. **Composite Loss Function Formulation**:
   - Define the total loss as $L = \lambda_1 L_{L1} + \lambda_2 L_{spec} + \lambda_3 L_{edge}$.
   - $L_{L1}$: Pixel-wise Mean Absolute Error between predicted and ground truth tSZ.
   - $L_{spec}$: Spectral consistency loss computed on apodized patches using `utils.powers`. Use a logarithmic scale for the loss to balance high-amplitude large-scale modes and low-amplitude small-scale modes (ell range 100–5000).
   - $L_{edge}$: Replace TV regularization with a Sobel-filter-based edge loss to preserve morphological features of gas pressure profiles without inducing "staircasing" artifacts.

4. **Training Strategy and Augmentation**:
   - Use the AdamW optimizer with a one-cycle learning rate scheduler.
   - Employ a fixed validation split (e.g., 150 patches) to ensure the model is evaluated on spatially distinct structures, preventing data leakage.
   - Augment training data with horizontal/vertical flips, 90-degree rotations, and small-scale Gaussian blurring to improve robustness to beam profile variations and noise characteristics.

5. **Evaluation Metrics and Null Testing**:
   - Compute RMSE and SSIM against ground truth tSZ maps.
   - Calculate the cross-correlation coefficient between predicted and ground truth tSZ as a function of multipole $\ell$.
   - Perform a "Null Test": run the model on pure noise realizations (no signal) to quantify and mitigate the hallucination of non-physical structures.

6. **Ablation Studies**:
   - Train a baseline model without CIB auxiliary maps to quantify the performance gain from the cross-attention mechanism.
   - Train a model without the spectral consistency loss ($L_{spec}$) to demonstrate its necessity in enforcing physical constraints and preventing feature hallucination.

7. **Inference and Uncertainty Quantification**:
   - Run the trained model on the held-out test set.
   - Apply Monte Carlo dropout at inference time to generate pixel-wise uncertainty maps, providing a confidence estimate for the reconstructed tSZ signal under varying noise conditions.