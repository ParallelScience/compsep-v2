# Guided Super-Resolution Denoising of Thermal Sunyaev-Zel'dovich Maps using a Conditional Diffusion Model

**Scientist:** denario-6 (Denario AI Research Scientist)
**Date:** 2026-04-16
**Best iteration:** 5

**[View Paper & Presentation](https://ParallelScience.github.io/compsep-v2/)**

## Abstract

Reconstructing high-resolution maps of the thermal Sunyaev-Zel'dovich (tSZ) effect, a crucial tracer of baryonic gas pressure, is fundamentally limited by instrumental noise and foreground contamination from the Cosmic Infrared Background (CIB). We introduce a deep learning framework that performs super-resolution denoising of tSZ maps from simulated multi-frequency observations of the FLAMINGO simulation, mimicking the upcoming Simons Observatory. Our approach utilizes a two-stage model: first, a U-Net-based Super-Resolution Denoising Autoencoder (SR-DAE) leverages high-frequency CIB maps to reconstruct 1-arcmin tSZ maps, guided by a composite loss function that ensures pixel-level accuracy and fidelity to the physical tSZ power spectrum. Second, this deterministic model is transitioned into a Conditional Diffusion Model (CDM) to provide robust, pixel-level uncertainty estimates. We demonstrate that our framework significantly outperforms standard linear component separation methods like constrained Internal Linear Combination and Wiener Filtering, achieving a substantial reduction in reconstruction error on the 1–5 arcmin scales critical for studying baryonic feedback. The model is robust to out-of-distribution tests, including extreme massive clusters and high-noise realizations, and yields a tighter integrated tSZ signal-mass scaling relation. The reconstructed power spectrum transfer function remains near unity across a broad range of angular scales, and the CDM-derived uncertainties are shown to be well-calibrated, providing a reliable measure of map fidelity for future cosmological analyses.
