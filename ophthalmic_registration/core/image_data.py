"""
Core data structures for ophthalmic image registration.

This module defines the fundamental data classes used throughout the
registration pipeline, including image containers, metadata structures,
and transform results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import numpy as np


class MotionModel(Enum):
    """Supported motion models for image registration."""
    TRANSLATION = "translation"      # 2 DOF: tx, ty
    EUCLIDEAN = "euclidean"          # 3 DOF: tx, ty, rotation
    AFFINE = "affine"                # 6 DOF: full affine
    HOMOGRAPHY = "homography"        # 8 DOF: perspective transform
    
    @property
    def opencv_flag(self) -> int:
        """Return the corresponding OpenCV motion type flag."""
        import cv2
        mapping = {
            MotionModel.TRANSLATION: cv2.MOTION_TRANSLATION,
            MotionModel.EUCLIDEAN: cv2.MOTION_EUCLIDEAN,
            MotionModel.AFFINE: cv2.MOTION_AFFINE,
            MotionModel.HOMOGRAPHY: cv2.MOTION_HOMOGRAPHY,
        }
        return mapping[self]
    
    @property
    def matrix_shape(self) -> Tuple[int, int]:
        """Return the expected transform matrix shape."""
        if self == MotionModel.HOMOGRAPHY:
            return (3, 3)
        return (2, 3)


class ImageModality(Enum):
    """Supported ophthalmic imaging modalities."""
    FUNDUS = "fundus"
    IR_ENFACE = "ir_enface"
    FAF = "faf"
    OCT_ENFACE = "oct_enface"
    ICGA = "icga"
    FA = "fluorescein_angiography"
    COLOR = "color"
    RED_FREE = "red_free"
    UNKNOWN = "unknown"


@dataclass
class PixelSpacing:
    """
    Represents physical pixel spacing in real-world units.
    
    Attributes:
        row_spacing: Physical distance between pixel centers along rows (mm)
        column_spacing: Physical distance between pixel centers along columns (mm)
        unit: Unit of measurement (default: 'mm')
        source: Source of the spacing information ('dicom', 'manual', 'estimated')
    """
    row_spacing: float
    column_spacing: float
    unit: str = "mm"
    source: str = "unknown"
    
    @property
    def is_isotropic(self) -> bool:
        """Check if pixel spacing is isotropic (equal in both directions)."""
        return np.isclose(self.row_spacing, self.column_spacing, rtol=1e-3)
    
    @property
    def mean_spacing(self) -> float:
        """Return mean pixel spacing for approximate calculations."""
        return (self.row_spacing + self.column_spacing) / 2.0
    
    def to_microns(self) -> 'PixelSpacing':
        """Convert spacing to microns."""
        if self.unit == "mm":
            return PixelSpacing(
                row_spacing=self.row_spacing * 1000,
                column_spacing=self.column_spacing * 1000,
                unit="um",
                source=self.source
            )
        elif self.unit == "um":
            return self
        else:
            raise ValueError(f"Cannot convert from unit '{self.unit}' to microns")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "row_spacing": self.row_spacing,
            "column_spacing": self.column_spacing,
            "unit": self.unit,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PixelSpacing':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ImageMetadata:
    """
    Comprehensive metadata container for ophthalmic images.
    
    Attributes:
        pixel_spacing: Physical pixel spacing
        modality: Imaging modality
        laterality: Eye laterality ('OD', 'OS', or None)
        acquisition_date: Date of image acquisition
        patient_id: Anonymized patient identifier
        study_uid: DICOM Study Instance UID
        series_uid: DICOM Series Instance UID
        instance_uid: DICOM SOP Instance UID
        manufacturer: Equipment manufacturer
        model_name: Equipment model name
        bits_stored: Bits per pixel stored
        bits_allocated: Bits per pixel allocated
        photometric_interpretation: DICOM photometric interpretation
        rows: Image height in pixels
        columns: Image width in pixels
        slice_thickness: Slice thickness for volumetric data (mm)
        image_orientation: Image orientation patient
        window_center: Display window center
        window_width: Display window width
        custom: Additional custom metadata
    """
    pixel_spacing: Optional[PixelSpacing] = None
    modality: ImageModality = ImageModality.UNKNOWN
    laterality: Optional[str] = None
    acquisition_date: Optional[datetime] = None
    patient_id: Optional[str] = None
    study_uid: Optional[str] = None
    series_uid: Optional[str] = None
    instance_uid: Optional[str] = None
    manufacturer: Optional[str] = None
    model_name: Optional[str] = None
    bits_stored: Optional[int] = None
    bits_allocated: Optional[int] = None
    photometric_interpretation: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    slice_thickness: Optional[float] = None
    image_orientation: Optional[List[float]] = None
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    custom: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_spatial_calibration(self) -> bool:
        """Check if spatial calibration information is available."""
        return self.pixel_spacing is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Handle modality being either enum or string
        if hasattr(self.modality, 'value'):
            modality_value = self.modality.value
        else:
            modality_value = self.modality
        
        # Handle acquisition_date being either datetime or string
        if self.acquisition_date:
            if hasattr(self.acquisition_date, 'isoformat'):
                acq_date = self.acquisition_date.isoformat()
            else:
                acq_date = self.acquisition_date  # Already a string
        else:
            acq_date = None
        
        result = {
            "modality": modality_value,
            "laterality": self.laterality,
            "acquisition_date": acq_date,
            "patient_id": self.patient_id,
            "study_uid": self.study_uid,
            "series_uid": self.series_uid,
            "instance_uid": self.instance_uid,
            "manufacturer": self.manufacturer,
            "model_name": self.model_name,
            "bits_stored": self.bits_stored,
            "bits_allocated": self.bits_allocated,
            "photometric_interpretation": self.photometric_interpretation,
            "rows": self.rows,
            "columns": self.columns,
            "slice_thickness": self.slice_thickness,
            "image_orientation": self.image_orientation,
            "window_center": self.window_center,
            "window_width": self.window_width,
            "custom": self.custom,
        }
        if self.pixel_spacing:
            # Handle pixel_spacing being either PixelSpacing object or dict
            if hasattr(self.pixel_spacing, 'to_dict'):
                result["pixel_spacing"] = self.pixel_spacing.to_dict()
            else:
                result["pixel_spacing"] = self.pixel_spacing  # Already a dict
        return result


@dataclass
class ImageData:
    """
    Container for an ophthalmic image with its metadata.
    
    This is the primary data structure passed through the registration pipeline.
    
    Attributes:
        pixel_array: Image pixel data as numpy array (H, W) or (H, W, C)
        metadata: Associated image metadata
        filepath: Original file path
        is_preprocessed: Flag indicating if preprocessing has been applied
        preprocessing_history: List of preprocessing steps applied
    """
    pixel_array: np.ndarray
    metadata: ImageMetadata = field(default_factory=ImageMetadata)
    filepath: Optional[str] = None
    is_preprocessed: bool = False
    preprocessing_history: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate image data after initialization."""
        if self.pixel_array is None:
            raise ValueError("pixel_array cannot be None")
        if self.pixel_array.ndim not in (2, 3):
            raise ValueError(f"pixel_array must be 2D or 3D, got {self.pixel_array.ndim}D")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return image shape."""
        return self.pixel_array.shape
    
    @property
    def height(self) -> int:
        """Return image height."""
        return self.pixel_array.shape[0]
    
    @property
    def width(self) -> int:
        """Return image width."""
        return self.pixel_array.shape[1]
    
    @property
    def is_color(self) -> bool:
        """Check if image is color (3 channels)."""
        return self.pixel_array.ndim == 3 and self.pixel_array.shape[2] == 3
    
    @property
    def is_grayscale(self) -> bool:
        """Check if image is grayscale."""
        return self.pixel_array.ndim == 2 or (
            self.pixel_array.ndim == 3 and self.pixel_array.shape[2] == 1
        )
    
    @property
    def dtype(self) -> np.dtype:
        """Return pixel array data type."""
        return self.pixel_array.dtype
    
    @property
    def pixel_spacing(self) -> Optional[PixelSpacing]:
        """Convenience accessor for pixel spacing."""
        return self.metadata.pixel_spacing
    
    def copy(self) -> 'ImageData':
        """Create a deep copy of the image data."""
        return ImageData(
            pixel_array=self.pixel_array.copy(),
            metadata=ImageMetadata(**self.metadata.to_dict()),
            filepath=self.filepath,
            is_preprocessed=self.is_preprocessed,
            preprocessing_history=self.preprocessing_history.copy()
        )
    
    def as_uint8(self) -> np.ndarray:
        """Return pixel array normalized to uint8 range."""
        if self.pixel_array.dtype == np.uint8:
            return self.pixel_array
        
        arr = self.pixel_array.astype(np.float64)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255
        else:
            arr = np.zeros_like(arr)
        return arr.astype(np.uint8)
    
    def as_float32(self) -> np.ndarray:
        """Return pixel array as float32 in [0, 1] range."""
        if self.pixel_array.dtype == np.float32:
            if self.pixel_array.max() <= 1.0:
                return self.pixel_array
        
        arr = self.pixel_array.astype(np.float32)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)
        return arr


@dataclass
class FeatureMatchResult:
    """
    Results from feature detection and matching.
    
    Attributes:
        keypoints_baseline: Keypoints detected in baseline image
        keypoints_followup: Keypoints detected in follow-up image
        descriptors_baseline: Descriptors for baseline keypoints
        descriptors_followup: Descriptors for follow-up keypoints
        matches: List of matched keypoint pairs
        inlier_mask: Boolean mask indicating RANSAC inliers
        num_inliers: Number of inlier matches
    """
    keypoints_baseline: List[Any]  # cv2.KeyPoint
    keypoints_followup: List[Any]  # cv2.KeyPoint
    descriptors_baseline: Optional[np.ndarray]
    descriptors_followup: Optional[np.ndarray]
    matches: List[Any]  # cv2.DMatch
    inlier_mask: Optional[np.ndarray] = None
    num_inliers: int = 0
    
    @property
    def match_ratio(self) -> float:
        """Ratio of inliers to total matches."""
        if len(self.matches) == 0:
            return 0.0
        return self.num_inliers / len(self.matches)


@dataclass
class TransformResult:
    """
    Complete result from image registration.
    
    Attributes:
        transform_matrix: The final transformation matrix
        motion_model: The motion model used
        coarse_transform: Transform from coarse (SIFT) alignment
        fine_transform: Transform from fine (ECC) alignment
        ecc_correlation: Final ECC correlation coefficient
        ecc_converged: Whether ECC optimization converged
        ecc_iterations: Number of ECC iterations performed
        feature_match_result: Feature matching details
        registration_time_ms: Total registration time in milliseconds
        quality_metrics: Dictionary of quality metrics
        warnings: List of warning messages
    """
    transform_matrix: np.ndarray
    motion_model: MotionModel
    coarse_transform: Optional[np.ndarray] = None
    fine_transform: Optional[np.ndarray] = None
    ecc_correlation: Optional[float] = None
    ecc_converged: bool = True
    ecc_iterations: int = 0
    feature_match_result: Optional[FeatureMatchResult] = None
    registration_time_ms: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_identity(self) -> bool:
        """Check if transform is approximately identity."""
        if self.motion_model == MotionModel.HOMOGRAPHY:
            identity = np.eye(3)
        else:
            identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        return np.allclose(self.transform_matrix, identity, atol=1e-6)
    
    @property
    def translation(self) -> Tuple[float, float]:
        """Extract translation component (tx, ty)."""
        if self.motion_model == MotionModel.HOMOGRAPHY:
            return (self.transform_matrix[0, 2], self.transform_matrix[1, 2])
        return (self.transform_matrix[0, 2], self.transform_matrix[1, 2])
    
    @property
    def rotation_degrees(self) -> Optional[float]:
        """Extract rotation angle in degrees (for Euclidean/Affine)."""
        if self.motion_model == MotionModel.TRANSLATION:
            return 0.0
        
        # Extract rotation from the 2x2 upper-left submatrix
        cos_theta = self.transform_matrix[0, 0]
        sin_theta = self.transform_matrix[1, 0]
        angle_rad = np.arctan2(sin_theta, cos_theta)
        return np.degrees(angle_rad)
    
    @property
    def scale_factors(self) -> Tuple[float, float]:
        """Extract scale factors (sx, sy) from affine/homography transform."""
        if self.motion_model in (MotionModel.TRANSLATION, MotionModel.EUCLIDEAN):
            return (1.0, 1.0)
        
        # Compute scale from SVD of the 2x2 rotation/scale matrix
        A = self.transform_matrix[:2, :2]
        U, S, Vt = np.linalg.svd(A)
        return (S[0], S[1])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "transform_matrix": self.transform_matrix.tolist(),
            "motion_model": self.motion_model.value,
            "coarse_transform": self.coarse_transform.tolist() if self.coarse_transform is not None else None,
            "fine_transform": self.fine_transform.tolist() if self.fine_transform is not None else None,
            "ecc_correlation": self.ecc_correlation,
            "ecc_converged": self.ecc_converged,
            "ecc_iterations": self.ecc_iterations,
            "registration_time_ms": self.registration_time_ms,
            "quality_metrics": self.quality_metrics,
            "warnings": self.warnings,
            "translation": self.translation,
            "rotation_degrees": self.rotation_degrees,
            "scale_factors": self.scale_factors,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransformResult':
        """Create from dictionary."""
        return cls(
            transform_matrix=np.array(data["transform_matrix"]),
            motion_model=MotionModel(data["motion_model"]),
            coarse_transform=np.array(data["coarse_transform"]) if data.get("coarse_transform") else None,
            fine_transform=np.array(data["fine_transform"]) if data.get("fine_transform") else None,
            ecc_correlation=data.get("ecc_correlation"),
            ecc_converged=data.get("ecc_converged", True),
            ecc_iterations=data.get("ecc_iterations", 0),
            registration_time_ms=data.get("registration_time_ms", 0.0),
            quality_metrics=data.get("quality_metrics", {}),
            warnings=data.get("warnings", []),
        )


@dataclass
class RegistrationConfig:
    """
    Configuration for the registration pipeline.
    
    Attributes:
        motion_model: Target motion model for registration
        use_coarse_alignment: Whether to use SIFT coarse alignment
        use_fine_alignment: Whether to use ECC fine alignment
        sift_n_features: Maximum number of SIFT features to detect
        sift_n_octave_layers: Number of octave layers for SIFT
        sift_contrast_threshold: Contrast threshold for SIFT
        sift_edge_threshold: Edge threshold for SIFT
        match_ratio_threshold: Lowe's ratio test threshold
        ransac_reproj_threshold: RANSAC reprojection threshold in pixels
        ransac_max_iters: Maximum RANSAC iterations
        ransac_confidence: RANSAC confidence level
        ecc_max_iterations: Maximum ECC iterations
        ecc_epsilon: ECC convergence epsilon
        ecc_gauss_filt_size: Gaussian filter size for ECC
        min_matches_required: Minimum matches for valid registration
        validate_transform: Whether to validate computed transforms
        max_translation_pixels: Maximum allowed translation for validation
        max_rotation_degrees: Maximum allowed rotation for validation
        max_scale_change: Maximum allowed scale change for validation
    """
    motion_model: MotionModel = MotionModel.AFFINE
    use_coarse_alignment: bool = True
    use_fine_alignment: bool = True
    
    # SIFT parameters
    sift_n_features: int = 5000
    sift_n_octave_layers: int = 3
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: float = 10
    
    # Matching parameters
    match_ratio_threshold: float = 0.75
    
    # RANSAC parameters
    ransac_reproj_threshold: float = 5.0
    ransac_max_iters: int = 2000
    ransac_confidence: float = 0.995
    
    # ECC parameters
    ecc_max_iterations: int = 1000
    ecc_epsilon: float = 1e-6
    ecc_gauss_filt_size: int = 5
    
    # Validation parameters
    min_matches_required: int = 10
    validate_transform: bool = True
    max_translation_pixels: float = 500.0
    max_rotation_degrees: float = 45.0
    max_scale_change: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "motion_model": self.motion_model.value,
            "use_coarse_alignment": self.use_coarse_alignment,
            "use_fine_alignment": self.use_fine_alignment,
            "sift_n_features": self.sift_n_features,
            "sift_n_octave_layers": self.sift_n_octave_layers,
            "sift_contrast_threshold": self.sift_contrast_threshold,
            "sift_edge_threshold": self.sift_edge_threshold,
            "match_ratio_threshold": self.match_ratio_threshold,
            "ransac_reproj_threshold": self.ransac_reproj_threshold,
            "ransac_max_iters": self.ransac_max_iters,
            "ransac_confidence": self.ransac_confidence,
            "ecc_max_iterations": self.ecc_max_iterations,
            "ecc_epsilon": self.ecc_epsilon,
            "ecc_gauss_filt_size": self.ecc_gauss_filt_size,
            "min_matches_required": self.min_matches_required,
            "validate_transform": self.validate_transform,
            "max_translation_pixels": self.max_translation_pixels,
            "max_rotation_degrees": self.max_rotation_degrees,
            "max_scale_change": self.max_scale_change,
        }
