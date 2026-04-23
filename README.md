# SMN: Subtractive Modulative Network with Learnable Periodic Activations

Official PyTorch implementation of **Subtractive Modulative Network (SMN)**, accepted at **IEEE ICASSP 2026**.

[Paper](docs/SMN_2601.pdf) | [Project Page](https://inrainbws.github.io/smn/) | [Supplementary Material](https://inrainbws.github.io/smn/)

SMN is a signal-processing-inspired implicit neural representation (INR) architecture. Instead of treating a coordinate MLP as a monolithic black box, SMN decomposes representation learning into a structured pipeline:

- **Oscillator**: a learnable periodic activation layer that generates a multi-frequency basis
- **Filter**: modulative mask modules that sculpt the spectrum and generate higher-order harmonics
- **Amplifier**: a lightweight self-mask stage that increases non-linearity without adding parameters

The same idea is used in this repository for:

- **2D image representation** on Kodak and DIV2K
- **3D novel view synthesis** in a NeRF pipeline

## Highlights

- Structured INR design inspired by subtractive synthesis rather than additive feature superposition
- Strong 2D reconstruction quality with **41.40 dB** on Kodak and **42.53 dB** on DIV2K
- Strong 3D generalization with **32.98 dB average PSNR** on the 8-scene synthetic NeRF benchmark
- Parameter-efficient model design with only **264,216 parameters** for the 2D SMN setting reported in the paper

## Results

The following numbers are reported in the paper.

### 2D Image Representation

| Method | Kodak PSNR | DIV2K PSNR | Parameters |
| --- | ---: | ---: | ---: |
| MLP | 28.63 | 30.21 | 272,415 |
| Gauss | 37.90 | 38.34 | 272,703 |
| SIREN | 33.65 | 33.73 | 272,703 |
| WIRE | 40.24 | 38.90 | 265,523 |
| RINR | 32.96 | 34.03 | 289,716 |
| **SMN (Ours)** | **41.40** | **42.53** | **264,216** |

### 3D NeRF Novel View Synthesis

Average PSNR on the 8 synthetic NeRF scenes at 400x400 resolution:

| Method | Avg. PSNR | Parameters |
| --- | ---: | ---: |
| PE + Gauss | 32.00 | 287,749 |
| PE + SIREN | 29.06 | 287,749 |
| PE + WIRE | 25.14 | 283,479 |
| PE + RINR | 26.84 | 314,703 |
| PE + MLP | 26.66 | 290,370 |
| **PE + SMN (Ours)** | **32.98** | **287,749** |

## Method Overview

SMN is designed around three ideas drawn from the paper:

1. **Learnable Oscillator**
   The first stage uses a learnable combination of sinusoidal bases instead of relying only on fixed positional encodings. This gives the network a more adaptive frequency basis for representing detailed signals.

2. **Modulative Filtering**
   The core filtering stages use multiplicative modulation, which the paper shows is more effective than simple additive combinations for harmonic generation and spectral sculpting.

3. **Self-Mask Amplifier**
   A final self-mask operation increases non-linearity and helps generate higher-order harmonics without increasing parameter count.

## Repository Structure

```text
SMN/
|-- docs/
|   `-- SMN_2601.pdf
|-- inr_base/
|   |-- app/           # 2D training scripts and utilities
|   |-- dataio/        # image and NeRF-style data loaders
|   |-- lib/           # INR / SMN model definitions
|   |-- codak/         # bundled Kodak images
|   `-- div2k/         # bundled DIV2K images
|-- nerf/
|   |-- configs/       # scene configs for NeRF experiments
|   |-- lib/           # auxiliary model variants and layers
|   |-- run_nerf.py    # main NeRF training / rendering entry point
|   `-- modelinr.py    # SMN-style backbone used in the NeRF pipeline
`-- README.md
```

## Installation

This repository contains research code for two related pipelines, so there is no single fully-pinned environment file for everything. A practical setup is:

```bash
pip install -r nerf/requirements.txt
pip install pandas scikit-image Pillow trimesh openpyxl
```

Notes:

- `nerf/requirements.txt` pins `torch==1.11.0`; if you install PyTorch separately for your CUDA version, adjust accordingly.
- The 2D pipeline exports PSNR values to Excel, so `openpyxl` is helpful.
- The 2D and 3D entry scripts rely on relative imports and paths. Run them from their respective subdirectories.

## Quick Start

### 2D Image Representation

Run the 2D pipeline from `inr_base/`:

```bash
cd inr_base
python app/train.py --data_dir div2k --output_dir outputs/div2k
```

Useful arguments:

- `--data_dir`: image directory, for example `div2k` or `codak`
- `--output_dir`: directory for reconstructed images
- `--image_size`: resized training resolution, default `768 512`
- `--n_adaptation`: number of images to process
- `--hidden_features`: hidden width of the INR backbone
- `--n_hidden_layers`: model depth
- `--inr_type`: INR type, default `siren`

Outputs:

- reconstructed images are written to `output_dir`
- PSNR values are saved to `psnrvaluesMLP.xlsx`

### 3D NeRF Synthesis

Run the NeRF pipeline from `nerf/`:

```bash
cd nerf
python run_nerf.py --config configs/lego.txt
```

Render from a trained checkpoint:

```bash
cd nerf
python run_nerf.py --config configs/lego.txt --render_only
```

Notes:

- scene configs live in `nerf/configs/*.txt`
- example synthetic data is expected under `nerf/data/nerf_synthetic/<scene_name>`
- outputs are written to `nerf/logs/<expname>/`

## Main Entry Points

If you want to understand or modify the core code, start here:

- `inr_base/app/train.py`: 2D training entry point
- `inr_base/lib/modelinr.py`: 2D SMN and INR backbone definitions
- `nerf/run_nerf.py`: 3D NeRF training and rendering entry point
- `nerf/modelinr.py`: SMN backbone used by the NeRF pipeline

## Acknowledgements

The NeRF pipeline builds on the excellent [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) implementation. Please also see `nerf/README.md` for the upstream workflow background.

## Citation

If you find this project useful, please cite the paper:

```bibtex
@misc{wang2026smn,
  title={Subtractive Modulative Network with Learnable Periodic Activations},
  author={Tiou Wang and Zhuoqian Yang and Markus Flierl and Mathieu Salzmann and Sabine Susstrunk},
  year={2026},
  note={IEEE ICASSP 2026}
}
```
