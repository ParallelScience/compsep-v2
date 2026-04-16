# Results and Discussion: Spectral-Spatial Super-Resolution Denoising for Baryonic Pressure Mapping

## 1. Validation Diagnostics and Data Characteristics

Before evaluating the performance of the Super-Resolution Denoising Autoencoder (SR-DAE), it is imperative to establish the baseline characteristics of the FLAMINGO L1_m9 HYDRO_FIDUCIAL simulation dataset and validate the evaluation pipeline. The physical objective of this study is to isolate the thermal Sunyaev-Zel'dovich (tSZ) effect—a direct tracer of the integrated line-of-sight electron pressure—from a complex mixture of the primary Cosmic Microwave Background (CMB), the kinetic Sunyaev-Zel'dovich (kSZ) effect, and the Cosmic Infrared Background (CIB). 

The evaluation pipeline was first rigorously validated. The <code>utils.powers</code> flat-sky angular power spectrum estimator was tested by computing the power spectrum of 20 realizations of Gaussian white noise. The computed average $C_\ell$ perfectly aligned with the theoretical expectation ($\Omega_{pix}$), confirming the accuracy of the pixel-to-physical mapping ($1.17$ arcmin/pixel) and ensuring that the spectral loss components and subsequent coherence metrics were mathematically sound. Furthermore, a coordinate alignment check was performed by computing the 2D cross-correlation between the ground-truth tSZ maps and their 90-degree rotated counterparts. As expected, the maximum absolute correlation coefficient was effectively zero, confirming the absence of systematic spatial offsets, phase-incoherence, or grid artifacts in the dataset geometry (visualized in <code>data/step_1_diagnostic_plot_1_1776331029.png</code>).

However, the numerical diagnostics of the ground-truth component maps revealed the extreme difficulty of the component separation task. The statistical analysis of the 1523 patches demonstrated a massive dynamic range disparity between the target signal and the contaminating foregrounds/backgrounds:
- **Lensed CMB**: Mean $0.89 \mu K_{CMB}$, Variance $1.25 \times 10^4$.
- **tSZ (Target)**: Mean $1.50 \times 10^{-6}$, Variance $3.28 \times 10^{-12}$ (dimensionless Compton-$y$).
- **kSZ**: Mean $2.89 \times 10^{-8}$, Variance $1.57 \times 10^{-12}$.
- **CIB (857 GHz)**: Mean $8.51 \times 10^5$ Jy/sr, Variance $3.35 \times 10^{10}$.

The tSZ signal is subdominant by several orders of magnitude. While the CMB dominates the variance at lower frequencies (90–150 GHz), the CIB completely overwhelms the signal at higher frequencies (353–857 GHz). The extreme sparsity and low amplitude of the tSZ signal—characterized by rare, high-intensity peaks corresponding to massive galaxy clusters embedded in a near-zero background of diffuse filaments—pose a severe optimization challenge. Although robust scaling (median and interquartile range) was applied to the input channels (saved in <code>data/scaling_params.npz</code>), the target tSZ maps were only soft-clipped at the 99.9th percentile ($2.17 \times 10^{-5}$) and not standardized to unit variance, a decision that profoundly impacted the model's learning dynamics.

## 2. Model Training Dynamics and the Illusion of Convergence

The SR-DAE model, featuring a multi-scale U-Net architecture with gated cross-attention designed to fuse Simons Observatory (SO) LAT bands with Planck HFI CIB channels, was trained on a stratified split of 1373 training patches and 150 validation patches. The training utilized the AdamW optimizer with a OneCycleLR scheduler over 30 epochs, employing mixed-precision (FP16) to manage the 56.8 million trainable parameters.

The training objective was a composite loss function comprising a pixel-wise L1 loss, a differentiable Haar wavelet loss, a pseudo-$C_\ell$ spectral loss, and a flux conservation loss (with its weight, $\lambda_4$, annealed from $0$ to $10^{-5}$ between epochs 5 and 20, as documented in <code>data/lambda_schedule.csv</code>).

On the surface, the training trajectory suggested successful optimization. The composite training loss decreased steadily from an initial value of $2.18 \times 10^{-1}$ to a final value of $3.09 \times 10^{-3}$. Similarly, the validation loss dropped from $1.29 \times 10^{-1}$ to $2.36 \times 10^{-3}$, with the best model weights saved at epoch 28 (<code>data/best_model.pth</code>).

However, this apparent convergence is an illusion driven by the pathological class imbalance (in a continuous sense) of the target tSZ maps. Because the vast majority of pixels in the tSZ ground truth represent the diffuse intergalactic medium or empty space with values on the order of $10^{-6}$, a model that collapses to predict a constant, near-zero value will achieve an exceptionally low L1 loss. The network optimized for the overwhelming majority of background pixels, effectively ignoring the rare, high-amplitude structural features (clusters) that carry the actual physical information regarding baryonic feedback. The auxiliary loss terms (wavelet and pseudo-$C_\ell$) were either weighted too conservatively to pull the model out of this local minimum or were similarly minimized by a flat, featureless prediction.

## 3. Signal-Injection Recovery: Quantifying Structural Failure

To rigorously test the model's ability to reconstruct physical structures independent of the complex cosmic backgrounds, a Signal-Injection Test was performed. Known Generalized Navarro-Frenk-White (GNFW) cluster profiles were injected into pure SO and Planck noise realizations at three distinct mass scales. The results, saved in <code>data/signal_injection_results.npz</code>, definitively expose the model's failure to reconstruct localized pressure profiles.

The integrated-$Y$ recovery fractions (the ratio of the predicted integrated Compton-$y$ to the true injected integrated Compton-$y$) were anomalously high and physically meaningless:
- **Low Mass** ($\theta_c=1.0$ arcmin, peak=$10^{-5}$): Recovery Fraction = $61699.18 \pm 17903.10$
- **Medium Mass** ($\theta_c=2.0$ arcmin, peak=$5 \times 10^{-5}$): Recovery Fraction = $5160.26 \pm 1497.80$
- **High Mass** ($\theta_c=4.0$ arcmin, peak=$10^{-4}$): Recovery Fraction = $715.43 \pm 209.06$

This massive overestimation is a direct consequence of the model's collapse. The network learned to predict a uniform, slightly positive background value (likely approximating the mean tSZ signal of the training set, $\sim 1.5 \times 10^{-6}$). When this constant background is integrated over the entire $256 \times 256$ pixel patch (65,536 pixels), it yields a total integrated $Y$ that dwarfs the true integrated $Y$ of the highly localized GNFW injection.

This hypothesis is confirmed by the radial profile residuals. For the high-mass cluster injection (peak amplitude $10^{-4}$), the mean residual at the central radial bin ($r=0.5$ arcmin) is $-1.33 \times 10^{-4}$. Because the residual is defined as Predicted - True, a residual of $-1.33 \times 10^{-4}$ against a true value of $10^{-4}$ implies that the model's prediction at the cluster core is effectively zero (or slightly negative). The model completely fails to reconstruct the peak of the cluster, acting instead as a strong attenuator that flattens all input structures into a uniform background.

## 4. Spectral Coherence: Complete Loss of Phase Information

To assess the spatial fidelity of the reconstructions, the cross-correlation coefficient $r_\ell$ was computed as a function of multipole $\ell$ on the held-out validation set. The coefficient is defined as:
$$ r_\ell = \frac{C_\ell^{PT}}{\sqrt{C_\ell^{PP} C_\ell^{TT}}} $$
where $C_\ell^{PT}$ is the cross-power spectrum between the predicted and true maps, and $C_\ell^{PP}$ and $C_\ell^{TT}$ are their respective auto-power spectra.

The results (stored in <code>data/performance_results.npz</code>) show that across the entire spatial frequency domain—from large scales ($\ell = 32.7$) down to the beam resolution limit ($\ell = 12476.7$)—the cross-correlation coefficient is exactly $0.0000$.

This indicates an absolute lack of phase coherence. The model has failed to learn any spatial structural information about the tSZ field. The spatial frequencies of the predictions are completely uncorrelated with the ground truth. In a successful super-resolution or component separation task, $r_\ell$ should approach $1.0$ at low multipoles and gradually decay at higher multipoles where instrumental noise dominates. An $r_\ell$ of zero across all scales is a definitive indicator of model collapse; the network is outputting maps that contain no spatial correlation with the actual distribution of ionized gas in the FLAMINGO simulation.

## 5. Bias-Variance Analysis: Systematic Attenuation

A detailed bias-variance check was performed to quantify the systematic errors as a function of the true tSZ signal intensity. The true tSZ pixels from the validation set were binned into 10 intervals, and the mean and standard deviation of the residuals were computed for each bin.

The statistics reveal a stark and systematic attenuation:
- In the lowest signal bin ($[-1.08 \times 10^{-6}, 7.30 \times 10^{-6})$), which contains the vast majority of the pixels (9,719,496 counts), the mean residual is $-1.85 \times 10^{-6}$.
- As the true tSZ signal increases, the mean residual becomes increasingly negative. For the medium-density bin ($[3.24 \times 10^{-5}, 4.08 \times 10^{-5})$), the mean residual is $-4.49 \times 10^{-5}$.
- In the highest signal bin, representing the cores of massive clusters ($[7.43 \times 10^{-5}, 8.26 \times 10^{-5}]$), the mean residual reaches $-1.27 \times 10^{-4}$.

The relationship between the true signal $T$ and the residual $R = P - T$ is approximately $R \approx -T$. This mathematically implies that the predicted value $P \approx 0$ across all signal regimes. The model systematically underestimates high-density regions (clusters) by an amount exactly equal to their amplitude, and similarly underestimates low-density regions (filaments).

Furthermore, the standard deviation of the residuals remains relatively constant across all bins (ranging from $6.89 \times 10^{-5}$ to $8.22 \times 10^{-5}$). This standard deviation is significantly larger than the mean tSZ signal itself, indicating that the variance in the residuals is entirely driven by the intrinsic variance of the true signal that the model is failing to capture. The network is not producing a noisy estimate of the truth; it is producing a flat map, leaving the entirety of the true signal's variance in the residual.

## 6. Interpretability and Feature Attribution

To understand the network's internal decision-making process, Integrated Gradients (IG) were employed to generate saliency maps, attributing the model's output to the SO (main) and Planck CIB (auxiliary) input channels. This analysis was performed on a representative high-mass cluster (patch index 109, peak tSZ $8.26 \times 10^{-5}$) and a low-mass filament (patch index 37, peak tSZ $5.47 \times 10^{-5}$), with the visualizations saved as <code>data/step_6_saliency_high_mass_cluster_1776334475.png</code> and <code>data/step_6_saliency_low_mass_filament_1776334476.png</code>.

The original hypothesis was that the gated cross-attention mechanism would allow the model to learn the spatial correlation between CIB-traced star-forming regions and tSZ-traced ionized gas flows, using the high-frequency CIB maps to guide the recovery of small-scale pressure fluctuations.

However, given the quantitative evidence of model collapse, the interpretability analysis yields a different conclusion. Because the model has learned to predict a near-zero constant to minimize the L1 loss against a highly sparse target, the gradients of the output with respect to the input features do not highlight physical structures. Instead, the saliency maps reflect the network's learned suppression mechanism. The attention gates likely learned to close, blocking the flow of information from both the main and auxiliary channels to prevent the high-variance CMB and CIB noise from perturbing the flat output. The network did not fail to find the correlation between the CIB and the tSZ; rather, the optimization landscape dictated that ignoring all inputs was the safest path to minimizing the global loss.

## 7. Synthesis and Future Directions

The objective of this research was to implement a Super-Resolution Denoising Autoencoder to reconstruct 1-arcmin resolution tSZ Compton-$y$ maps from noisy, beam-smeared SO and Planck observations. The results demonstrate that the current architecture and training paradigm are ineffective at disentangling the non-Gaussian tSZ signal from the dominant Gaussian CMB and CIB backgrounds.

The root cause of this failure is not necessarily architectural, but rather stems from the extreme amplitude disparity and sparsity of the target variable. The variance of the tSZ signal is up to 22 orders of magnitude smaller than the variance of the high-frequency CIB channels. While the input features were robustly scaled, the target tSZ maps were left in their native, highly sparse physical units. Consequently, the L1 reconstruction loss was overwhelmingly dominated by the empty background pixels. The network quickly discovered a pathological local minimum: predicting a featureless, near-zero map yields a highly "competitive" loss score, bypassing the need to learn complex, non-linear component separation filters.

To successfully map baryonic pressure using deep learning on datasets like the FLAMINGO simulation, future iterations of this work must address this optimization pathology directly. We recommend the following structural changes to the methodology:

1. **Target Normalization**: The ground-truth tSZ maps must be standardized to zero mean and unit variance during training. This ensures that the gradients backpropagated from the loss function are of a sufficient magnitude to update the network weights meaningfully, preventing the signal from being treated as numerical noise.
2. **Loss Function Redesign**: A standard L1 or MSE loss is insufficient for highly sparse, non-Gaussian fields. The implementation of a Focal Loss or a heavily weighted MSE—where errors in high-amplitude pixels (clusters) are penalized exponentially more than errors in background pixels—is required to force the network to attend to physical structures.
3. **Physical Preprocessing**: Expecting a neural network to implicitly learn to subtract the primary CMB (which is orders of magnitude brighter than the tSZ) is inefficient. Applying a standard Internal Linear Combination (ILC) to the input maps to null the primary CMB before feeding the residuals into the autoencoder would drastically reduce the dynamic range the network must navigate, allowing the convolutional filters to focus exclusively on separating the tSZ from the CIB and instrumental noise.

In conclusion, while the multi-scale U-Net with gated cross-attention represents a theoretically sound approach to multi-frequency data fusion, its application to CMB component separation requires rigorous conditioning of the target space and specialized loss landscapes to overcome the inherent sparsity of baryonic feedback signatures.