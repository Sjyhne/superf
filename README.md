# SuperF: Neural Implicit Fields for Multi-Image Super-Resolution

[![arXiv](https://img.shields.io/badge/arXiv-2512.09115-b31b1b.svg)](https://arxiv.org/abs/2512.09115)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://sjyhne.github.io/superf/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

A test-time optimization approach for multi-image super-resolution (MISR) that leverages coordinate-based neural networks (implicit neural representations). SuperF reconstructs high-resolution imagery from multiple shifted low-resolution frames **without requiring high-resolution training data**.

## Key Features

- **Joint Optimization**: Shares a single INR across multiple low-resolution frames while simultaneously optimizing sub-pixel frame alignment
- **Affine Transformation Parameterization**: Directly parameterizes alignment as optimizable affine transformation parameters
- **Super-sampled Coordinate Grid**: Optimizes on a grid corresponding to the desired output resolution
- **Uncertainty Estimation**: Learns pixel-wise confidence maps to identify and downweight unreliable pixels (e.g., clouds in satellite imagery)
- **Training-Data Independent**: Operates without requiring high-resolution training datasets
- **Versatile**: Handles both satellite imagery and handheld camera bursts with upsampling factors up to 8x

## Installation

```bash
# Clone the repository
git clone https://github.com/sjyhne/superf.git
cd superf

# Install dependencies (recommended: use a virtual environment)
pip install -e .

# Or install with development tools
pip install -e ".[dev]"
```

**Requirements**: Python 3.10+, PyTorch 2.0+, CUDA-capable GPU recommended

**Optional — Handheld bursts**: The script `run_handheld.py` provides super-resolution for handheld camera bursts. It requires the separate `handheld` Python package (not included in this repo). The main entry point `optimize.py` does not depend on `handheld` and works for satellite and synthetic data only.

## Quick Start

### Generate Synthetic Training Data

Create synthetic multi-frame data from a single image:

```bash
python create_data_from_single_image.py --input_image path/to/image.png --output_dir data/sample_1
```

### Run Super-Resolution

```bash
# Basic training with default settings (4x upsampling)
python optimize.py --dataset satburst_synth --sample_id sample_1 --df 4 --iters 1000

# With uncertainty estimation
python optimize.py --dataset satburst_synth --sample_id sample_1 --df 4 --iters 2000 \
    --use_gnll --use_direct_gnll --variance_reg 0.001

# Different model architectures
python optimize.py --model siren --input_projection fourier_10 --df 4 --iters 2000
python optimize.py --model wire --input_projection fourier_10 --df 4 --iters 2000
```

### Training on Real Satellite Data (WorldStrat)

```bash
# Basic WorldStrat training
python optimize.py --dataset worldstrat --root_worldstrat ~/data/worldstrat_kaggle/ \
    --area_name "Landcover-1295513" --df 4 --num_samples 8 --iters 10000

# With affine transformation (translation + rotation)
python optimize.py --dataset worldstrat --root_worldstrat ~/data/worldstrat_kaggle/ \
    --area_name "UNHCR-SYRs008164" --df 4 --num_samples 8 --iters 10000 --rotation True
```

## Method Overview

SuperF learns sub-pixel transformations from multiple low-resolution observations to reconstruct a high-resolution image:

1. **Input**: Multiple LR frames with unknown sub-pixel shifts
2. **Optimization**: Jointly optimize an INR and per-frame affine transformations
3. **Uncertainty**: Learn pixel-wise confidence to handle clouds, shadows, and moving objects
4. **Output**: Super-resolved image at the target resolution

### Data Generation Flow

For synthetic experiments, training data is generated as follows:

1. Start with an HR image (e.g., 256x256)
2. Upsample 2x using bilinear interpolation (512x512)
3. Apply 16 random sub-pixel translations (-6 to +6 pixels)
4. Create HR-LR pairs by downsampling each translated version
5. Track all transformation parameters for evaluation

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset: `satburst_synth`, `worldstrat`, `burst_synth`, `worldstrat_test` | `satburst_synth` |
| `--sample_id` | Sample identifier | `Landcover-743192_rgb` |
| `--df` | Downsampling factor (2, 4, or 8) | `4` |
| `--num_samples` | Number of LR frames | `16` |
| `--model` | Model: `mlp`, `siren`, `wire`, `linear`, `conv`, `thera`, `nir` | `mlp` |
| `--input_projection` | Projection: `fourier_10`, `fourier_20`, `legendre`, `linear`, `none` | `fourier_10` |
| `--iters` | Training iterations | `1000` |
| `--use_gnll` | Enable Gaussian NLL for uncertainty | `False` |
| `--rotation` | Optimize rotation angle | `False` |
| `--aug` | Augmentation: `none`, `light`, `medium`, `heavy` | `none` |

## Results Structure

After training, results are organized as:

```
results/
└── {dataset}/
    └── df{factor}_shift{shift}_samples{n}/
        └── {sample_id}/
            └── {model}_{projection}_{iters}/
                ├── comparison.png
                ├── final_training_curves.png
                ├── final_translation_vis.png
                ├── metrics.json
                ├── output_prediction.png
                ├── hr_ground_truth.png
                └── lr_input.png
```

## Visualization

The project includes visualization utilities in `viz_utils.py`:

```python
from viz_utils import plot_training_curves, visualize_translations, create_model_comparison_grid

# Plot training curves
plot_training_curves(history, save_path='training_curves.png')

# Visualize predicted vs ground truth translations
visualize_translations(pred_dx, pred_dy, target_dx, target_dy, save_path='translations.png')

# Compare multiple models
create_model_comparison_grid(base_dir='results', save_dir='comparison')
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{jyhne2025superf,
  title={SuperF: Neural Implicit Fields for Multi-Image Super-Resolution},
  author={Jyhne, Sander Riisøen and Igel, Christian and Goodwin, Morten and
          Andersen, Per-Arne and Belongie, Serge and Lang, Nico},
  journal={arXiv preprint arXiv:2512.09115},
  year={2025}
}
```

## Authors

- **Sander Riisøen Jyhne** - University of Agder
- **Christian Igel** - University of Copenhagen
- **Morten Goodwin** - University of Agder
- **Per-Arne Andersen** - University of Agder
- **Serge Belongie** - University of Copenhagen
- **Nico Lang** - University of Copenhagen

## Acknowledgments

This work was funded by the Pioneer Centre for AI (DNRF grant P1) and the Global Wetland Center (NNF23OC0081089) from Novo Nordisk Foundation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
