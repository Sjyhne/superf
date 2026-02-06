# SuperF: Neural Implicit Fields for Multi-Image Super-Resolution

[![arXiv](https://img.shields.io/badge/arXiv-2512.09115-b31b1b.svg)](https://arxiv.org/abs/2512.09115)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://sjyhne.github.io/superf/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A test-time optimization approach for multi-image super-resolution (MISR) that leverages coordinate-based neural networks (implicit neural representations). SuperF reconstructs high-resolution imagery from multiple shifted low-resolution frames **without requiring high-resolution training data**.

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

## Quick Start

### Run Super-Resolution

```bash
# Basic training with default settings (4x upsampling)
python optimize.py --dataset satburst_synth --sample_id sample_1 --df 4 --iters 1000

# With uncertainty estimation
python optimize.py --dataset satburst_synth --sample_id sample_1 --df 4 --iters 2000 \
    --use_gnll --variance_reg 0.001

# Alternative model (NIR)
python optimize.py --model nir --input_projection fourier_10 --df 4 --iters 2000
```

### Complete example (all arguments)

A single fully specified run using every argument (synthetic data; ensure `data/sample_1` exists, e.g. via `create_data_from_single_image.py`):

```bash
python optimize.py \
  --dataset satburst_synth \
  --sample_id sample_1 \
  --df 4 \
  --scale_factor 4 \
  --lr_shift 1.0 \
  --num_samples 16 \
  --aug none \
  --model mlp \
  --network_depth 4 \
  --network_hidden_dim 256 \
  --projection_dim 256 \
  --input_projection fourier_10 \
  --fourier_scale 10.0 \
  --use_gnll \
  --use_separate_ud \
  --variance_reg 0.001 \
  --variance_smooth_reg 0.0001 \
  --visualize_variance \
  --iters 2000 \
  --learning_rate 2e-3 \
  --weight_decay 0.05 \
  --seed 6 \
  --device 0
```

Optional flags you can add when needed: `--no_base_frame`, `--no_direct_param_T`, `--use_color_shift`, `--no_variance_viz`. For multi-sample runs add `--multi_sample` and `--output_folder <dir>`.

### Training on Real Satellite Data (WorldStrat)

```bash
# Basic WorldStrat training
python optimize.py --dataset worldstrat --root_worldstrat ~/data/worldstrat_kaggle/ \
    --area_name "Landcover-1295513" --df 4 --num_samples 8 --iters 10000

# With affine transformation (translation + rotation)
python optimize.py --dataset worldstrat --root_worldstrat ~/data/worldstrat_kaggle/ \
    --area_name "UNHCR-SYRs008164" --df 4 --num_samples 8 --iters 10000 --rotation True
```

### Extract from FORCE Datacube

If your data is in a [FORCE](https://force-eo.readthedocs.io/) datacube (Landsat/Sentinel-2 Level 2 or 3, organised by tiles like `X0134_Y0097`), you can extract RGB images from one tile and then run super-resolution on that folder.

**Step 1 — Extract**  
`extract_force_datacube.py` finds reflectance GeoTIFFs (BOA or TOA) in the given tile, reads the first three bands as RGB, scales to 0–255, and writes PNGs plus a short manifest. Each date/sensor becomes one image; multiple images in the folder act as the multi-frame input for SuperF.

```bash
# Required: path to datacube root and tile ID
python extract_force_datacube.py --datacube /path/to/level2 --tile X0134_Y0097 --output_dir force_extract

# Optional: limit to a date range and cap the number of images
python extract_force_datacube.py --datacube /path/to/level2 --tile X0134_Y0097 --output_dir force_extract \
    --date_from 20200101 --date_to 20201231 --max_files 16

# Optional: only certain sensors (e.g. Sentinel-2A and Landsat 8)
python extract_force_datacube.py --datacube /path/to/level2 --tile X0134_Y0097 --output_dir force_extract \
    --sensors SEN2A LND08 --max_files 12
```

**Step 2 — Super-resolution**  
Point `optimize_against_folder.py` at the extracted folder. It will use all PNGs in that folder as LR frames and output a super-resolved result.

```bash
python optimize_against_folder.py --input_folder force_extract --output_folder sr_results --df 2
```

**Useful options for `extract_force_datacube.py`:**

| Option | Description |
|--------|-------------|
| `--datacube` | Path to FORCE datacube root (e.g. directory containing tile folders). |
| `--tile` | Tile ID (e.g. `X0134_Y0097`). |
| `--output_dir` | Where to write PNGs (default: `force_extract`). |
| `--product` | `BOA` (default) or `TOA`. |
| `--date_from` / `--date_to` | Filter by acquisition date (`YYYYMMDD`). |
| `--sensors` | Only include these sensor IDs (e.g. `SEN2A`, `LND08`). |
| `--max_files` | Maximum number of images to extract. |
| `--band_indices` | 0-based R,G,B band indices (default: `0 1 2`). |

Run `python extract_force_datacube.py --help` for the full list.

### Generate Synthetic Training Data

Create synthetic multi-frame data from a single image:

```bash
python create_data_from_single_image.py --input_image path/to/image.png --output_dir data/sample_1
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
| `--model` | Model: `mlp`, `nir` | `mlp` |
| `--input_projection` | Fourier projection scale: `fourier_10`, `fourier_5`, `fourier_20`, `fourier_40`, `fourier` | `fourier_10` |
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
