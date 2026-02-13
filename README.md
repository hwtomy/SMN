# Subtractive Modulative Network (SMN) with Learnable Periodic Activations

Official implementation of **“Subtractive Modulative Network with Learnable Periodic Activations”** (IEEE ICASSP 2026).

## Resources
- **Paper (PDF)**: `docs/SMN_2601.pdf` 
- **Project Page**: https://inrainbws.github.io/smn/
- **Supplementary Materials**: https://inrainbws.github.io/smn/
- **ICASSP 2026 Accepted Papers** (search by title): https://cmsworkshops.com/ICASSP2026/papers/accepted_papers.php

## Abstract
We propose the **Subtractive Modulative Network (SMN)**, a parameter-efficient Implicit Neural Representation (INR) architecture inspired by **subtractive synthesis**. SMN is structured as a signal-processing pipeline with (i) an **Oscillator**—a learnable periodic activation layer that generates a multi-frequency basis—and (ii) **Filters**—modulative mask modules that generate high-order harmonics. We provide theoretical analysis and empirical validation, achieving **40+ dB PSNR** on two image datasets and showing consistent advantages on **3D NeRF novel view synthesis**.

## Quick Start

### 2D Image Representation (Kodak / DIV2K)
```bash
python inr_base/app/train.py

```

### 3D NeRF Synthesis (e.g., Lego)

```bash
python nerf/run_nerf.py --config configs/nerf_lego.txt
```




