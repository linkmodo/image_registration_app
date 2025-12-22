# Ophthalmic Image Registration Application

A professional Python application for longitudinal ophthalmic image registration and comparison, supporting DICOM and standard image formats with multiple alignment methods.

**Version 2.3.0** | Updated: December 2025

## Overview

This application provides robust image registration with multiple alignment options:

### Alignment Methods
- **AKAZE**: Fast and accurate feature-based alignment (recommended default)
- **ORB**: Fastest option for quick previews
- **SIFT**: Classic algorithm for detailed images
- **Manual 3-Point**: For difficult cases where automatic methods fail

### Supported Modalities
- Fundus photography
- IR (Infrared) en face images
- FAF (Fundus Autofluorescence) images
- Cross-modality registration (IR↔FAF) with CLAHE preprocessing
- Other planar retinal modalities

## Features

- **DICOM Support**: Full DICOM loading with metadata preservation
- **Pixel Spacing**: Accurate mm measurements from DICOM tags
- **Measurement Unit Toggle**: Switch between mm and pixels for measurements
- **CLAHE Preprocessing**: Contrast Limited Adaptive Histogram Equalization for cross-modality registration
- **Multimodality Options**: Grayscale, invert, contrast, brightness adjustments
- **Batch Processing**: Register multiple follow-up images at once
- **Batch Export**: Export all registered images to DICOM, PNG, TIFF, or JPEG
- **Reference Baseline**: Lock baseline for longitudinal studies
- **Measurement Tools**: Distance and area measurements via right-click menu
- **Registration Points Visualization**: Up to 50 spatially distributed points with connecting lines
- **Multiple Comparison Modes**: Side-by-side, Overlay, Difference, Checkerboard, Split View

## Architecture

```
ophthalmic_registration/
├── core/
│   ├── image_data.py          # Image data structures and metadata
│   └── exceptions.py          # Custom exceptions
├── io/
│   ├── dicom_loader.py        # DICOM image loading with metadata
│   ├── standard_loader.py     # Standard format loading (PNG, TIFF, etc.)
│   └── image_io.py            # Unified image I/O interface
├── preprocessing/
│   └── pipeline.py            # Preprocessing pipeline (CLAHE, etc.)
├── registration/
│   ├── feature_aligner.py     # Feature-based alignment (AKAZE, ORB, SIFT)
│   ├── sift_aligner.py        # SIFT-specific alignment
│   └── registration_pipeline.py  # Unified registration pipeline
├── measurement/
│   └── spatial.py             # Spatial calibration and measurements
├── visualization/
│   └── comparison.py          # Visualization and comparison tools
├── export/
│   └── output.py              # Export utilities (DICOM, PNG, TIFF, JPEG)
├── gui/
│   ├── main_window.py         # Main application window
│   ├── image_viewer.py        # Image viewer with zoom/pan/measurements
│   ├── comparison_view.py     # Multi-mode comparison widget
│   ├── controls_panel.py      # Registration settings panel
│   ├── batch_registration.py  # Batch registration dialog
│   ├── batch_export_dialog.py # Batch export dialog
│   ├── manual_registration.py # Manual 3-point registration
│   └── styles.py              # Modern dark/light themes
└── utils/
    └── logging_config.py      # Logging configuration
```

## Installation

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For additional preprocessing options:
```bash
pip install scikit-image  # For advanced filters
```

## Quick Start

### GUI Application

Launch the graphical user interface:

```bash
python run_gui.py
```

### Basic Workflow

1. **Load Baseline**: File → Open Baseline (Ctrl+O)
2. **Load Follow-up**: File → Open Follow-up (Ctrl+Shift+O)
3. **Register**: Click "Register" button or press Ctrl+R
4. **Compare**: Switch to "Compare" tab to view results
5. **Export**: File → Export Results (Ctrl+E)

### Measurement Tools

Right-click on any image to access measurement tools:
- **Measure → Distance**: Click two points to measure distance
- **Measure → Area**: Click multiple points, double-click to complete polygon
- Measurements display in mm if DICOM pixel spacing is available

### Batch Processing

1. Load baseline image
2. Registration → Batch Registration (Ctrl+B)
3. Select multiple follow-up images
4. Process all at once

### Batch Export

1. Register one or more follow-up images
2. Registration → Batch Export Registered Images (Ctrl+Shift+E)
3. Choose format (DICOM, PNG, TIFF, JPEG)
4. Select additional outputs (overlays, difference maps, reports)

### Python API

```python
from ophthalmic_registration import RegistrationPipeline, ImageLoader

# Load images
loader = ImageLoader()
baseline = loader.load("baseline_fundus.dcm")
followup = loader.load("followup_fundus.dcm")

# Register images
pipeline = RegistrationPipeline()
result, registered = pipeline.register_and_apply(baseline, followup)

# Export
from ophthalmic_registration import ExportManager
exporter = ExportManager(output_dir="./results")
exporter.export_registration_results(baseline, followup, registered, result)
```

## Modules

### Image I/O (`io/`)
- Unified loading for DICOM and standard formats
- Automatic metadata extraction (pixel spacing, orientation)
- Intensity normalization for consistent processing

### Preprocessing (`preprocessing/`)
- Grayscale conversion
- CLAHE contrast enhancement
- Vessel enhancement filters
- Resolution normalization

### Registration (`registration/`)
- **Feature-based**: AKAZE, ORB, SIFT with RANSAC
- Supports: Translation, Euclidean, Affine, Homography motion models
- CLAHE preprocessing for cross-modality alignment

### Measurement (`measurement/`)
- Pixel-to-mm conversion using DICOM pixel spacing
- Distance and area measurements
- Displayed directly on image viewer

### Export (`export/`)
- DICOM export with metadata preservation
- Lossless formats: PNG, TIFF
- Lossy format: JPEG with quality control
- Overlay images, difference maps, transform JSON, reports

## License

For research and clinical evaluation purposes.

## Author

Created by Li Fan, 2025

## References

- AKAZE: Alcantarilla, P.F. et al. (2013). Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces
- CLAHE: Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization
