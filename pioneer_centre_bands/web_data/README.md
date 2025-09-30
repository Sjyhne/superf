# Super-Resolution Optimization Visualization Data

## 🎯 Quick Start Guide

### Loading Data in JavaScript:
```javascript
// Load all data files
const masterData = await fetch('web_data/master_data.json').then(r => r.json());
const lossData = await fetch('web_data/loss_evolution.json').then(r => r.json());
const alignmentData = await fetch('web_data/alignment_evolution.json').then(r => r.json());
const imageData = await fetch('web_data/image_evolution.json').then(r => r.json());

// Get basic experiment info
const totalIterations = masterData.experiment_info.total_iterations;
const numSamples = masterData.experiment_info.num_samples;
const checkpointIterations = alignmentData.iterations; // [0, 50, 100, 150, ...]
```

## 📊 Data Access Patterns

### 1. Image Display:
```javascript
// Show image for sample 0 at iteration 100 in web size
function showImage(sampleId, iterationNumber, method = 'sr', size = 'web') {
    const iterationIndex = imageData.iterations.indexOf(iterationNumber);
    if (iterationIndex === -1) {
        console.error('Iteration not available:', iterationNumber);
        return;
    }
    
    const imagePath = imageData.image_paths[`sample_${sampleId}`][`${method}_progression`][size][iterationIndex];
    document.getElementById('main-image').src = imagePath;
}

// Get LR reference image
function showReference(sampleId, size = 'web') {
    const imagePath = imageData.image_paths[`sample_${sampleId}`].lr_reference[size];
    document.getElementById('reference-image').src = imagePath;
}

// Show bilinear baseline
function showBilinear(sampleId, iterationNumber, size = 'web') {
    showImage(sampleId, iterationNumber, 'bilinear', size);
}
```

### 2. Loss Visualization:
```javascript
// Plot loss curves (handle null values for iteration 0)
function plotLosses() {
    const iterations = lossData.iterations;
    const reconLosses = lossData.recon_losses.map(v => v === null ? undefined : v);
    const transLosses = lossData.trans_losses.map(v => v === null ? undefined : v);
    
    // Use your preferred charting library with spanGaps: true for null values
}
```

### 3. Timeline Control:
```javascript
// Create interactive timeline
function createTimeline() {
    const iterations = alignmentData.iterations;
    const slider = document.getElementById('iteration-slider');
    
    slider.min = 0;
    slider.max = iterations.length - 1;
    slider.value = 0;
    
    slider.addEventListener('input', (e) => {
        const iterationIndex = parseInt(e.target.value);
        const iterationNumber = iterations[iterationIndex];
        updateAllVisualization(iterationNumber);
    });
}
```

## ⚠️ Important Notes:

### Null Values:
- **Iteration 0 losses**: `null` (no training occurred yet)
- Always check for `null` before using loss values
- Use `spanGaps: true` in charts to handle null values

### Image Sizes:
- **Thumbnail (200×200)**: Timeline previews, overview grids
- **Web (800×800)**: Main display, takes up most screen space  
- **Fullscreen (1200×1200)**: Modal/fullscreen viewing

### Available Methods:
- **sr**: Super-resolved output from the model
- **lr**: Original low-resolution reference
- **bilinear**: Bilinear interpolation baseline

### Array Indexing:
- Iteration numbers: `[0, 50, 100, 150, ...]` (actual iteration values)
- Array indices: `[0, 1, 2, 3, ...]` (for accessing arrays)
- Always use `iterations.indexOf(iterationNumber)` to convert

## 📈 Sample Data Summary:
- **Training iterations**: 0 → 5000
- **Checkpoints saved**: 2 (including initial state)
- **Samples per checkpoint**: 13
- **Total images**: ~234 (3 methods × 3 sizes × samples × checkpoints)
- **No infinity values**: All numeric data is finite or explicitly null

Generated on: 2025-09-30T07:30:37.919225
