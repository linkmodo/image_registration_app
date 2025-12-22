"""
Ophthalmic Image Registration Application

A modular Python application for longitudinal ophthalmic image registration
and comparison using OpenCV and medical imaging best practices.

Supports:
- DICOM and standard image formats
- Two-stage registration (SIFT coarse + ECC fine)
- Spatial calibration with real-world measurements
- Comprehensive visualization and comparison tools
"""

__version__ = "1.0.0"
__author__ = "Medical Imaging Engineering Team"

from ophthalmic_registration.core.image_data import (
    ImageData,
    ImageMetadata,
    PixelSpacing,
    TransformResult,
)
from ophthalmic_registration.core.exceptions import (
    RegistrationError,
    ImageLoadError,
    PreprocessingError,
    MetadataError,
)
from ophthalmic_registration.io.image_io import ImageLoader
from ophthalmic_registration.preprocessing.pipeline import PreprocessingPipeline
from ophthalmic_registration.registration.registration_pipeline import RegistrationPipeline
from ophthalmic_registration.visualization.comparison import Visualizer
from ophthalmic_registration.measurement.spatial import SpatialCalibration
from ophthalmic_registration.export.output import ExportManager

__all__ = [
    # Core data structures
    "ImageData",
    "ImageMetadata",
    "PixelSpacing",
    "TransformResult",
    # Exceptions
    "RegistrationError",
    "ImageLoadError",
    "PreprocessingError",
    "MetadataError",
    # Main components
    "ImageLoader",
    "PreprocessingPipeline",
    "RegistrationPipeline",
    "Visualizer",
    "SpatialCalibration",
    "ExportManager",
]
