"""Core data structures and exceptions for ophthalmic image registration."""

from ophthalmic_registration.core.image_data import (
    ImageData,
    ImageMetadata,
    PixelSpacing,
    TransformResult,
    MotionModel,
)
from ophthalmic_registration.core.exceptions import (
    RegistrationError,
    ImageLoadError,
    PreprocessingError,
    MetadataError,
    CalibrationError,
)

__all__ = [
    "ImageData",
    "ImageMetadata",
    "PixelSpacing",
    "TransformResult",
    "MotionModel",
    "RegistrationError",
    "ImageLoadError",
    "PreprocessingError",
    "MetadataError",
    "CalibrationError",
]
