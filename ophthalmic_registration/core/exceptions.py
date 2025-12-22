"""
Custom exceptions for the ophthalmic image registration application.

This module defines a hierarchy of exceptions for clear error handling
and reporting throughout the registration pipeline.
"""

from typing import Optional


class OphthalmicRegistrationError(Exception):
    """Base exception for all ophthalmic registration errors."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.full_message)
    
    @property
    def full_message(self) -> str:
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class ImageLoadError(OphthalmicRegistrationError):
    """Raised when image loading fails."""
    
    def __init__(self, filepath: str, reason: str, details: Optional[str] = None):
        self.filepath = filepath
        message = f"Failed to load image '{filepath}': {reason}"
        super().__init__(message, details)


class MetadataError(OphthalmicRegistrationError):
    """Raised when required metadata is missing or invalid."""
    
    def __init__(self, field: str, reason: str, details: Optional[str] = None):
        self.field = field
        message = f"Metadata error for '{field}': {reason}"
        super().__init__(message, details)


class PreprocessingError(OphthalmicRegistrationError):
    """Raised when preprocessing fails."""
    
    def __init__(self, step: str, reason: str, details: Optional[str] = None):
        self.step = step
        message = f"Preprocessing failed at step '{step}': {reason}"
        super().__init__(message, details)


class RegistrationError(OphthalmicRegistrationError):
    """Raised when registration fails."""
    
    def __init__(self, stage: str, reason: str, details: Optional[str] = None):
        self.stage = stage
        message = f"Registration failed at stage '{stage}': {reason}"
        super().__init__(message, details)


class CalibrationError(OphthalmicRegistrationError):
    """Raised when spatial calibration fails or is inconsistent."""
    
    def __init__(self, reason: str, details: Optional[str] = None):
        message = f"Calibration error: {reason}"
        super().__init__(message, details)


class TransformValidationError(RegistrationError):
    """Raised when a computed transform fails validation checks."""
    
    def __init__(self, reason: str, transform_type: str, details: Optional[str] = None):
        self.transform_type = transform_type
        super().__init__(
            stage=f"transform_validation ({transform_type})",
            reason=reason,
            details=details
        )


class ConvergenceError(RegistrationError):
    """Raised when iterative registration fails to converge."""
    
    def __init__(self, algorithm: str, iterations: int, details: Optional[str] = None):
        self.algorithm = algorithm
        self.iterations = iterations
        super().__init__(
            stage=algorithm,
            reason=f"Failed to converge after {iterations} iterations",
            details=details
        )


class InsufficientFeaturesError(RegistrationError):
    """Raised when insufficient features are detected for registration."""
    
    def __init__(
        self,
        detected: int,
        required: int,
        image_name: str = "unknown",
        details: Optional[str] = None
    ):
        self.detected = detected
        self.required = required
        super().__init__(
            stage="feature_detection",
            reason=f"Insufficient features in '{image_name}': {detected} detected, {required} required",
            details=details
        )


class MatchingError(RegistrationError):
    """Raised when feature matching fails."""
    
    def __init__(self, matches_found: int, required: int, details: Optional[str] = None):
        self.matches_found = matches_found
        self.required = required
        super().__init__(
            stage="feature_matching",
            reason=f"Insufficient matches: {matches_found} found, {required} required",
            details=details
        )
