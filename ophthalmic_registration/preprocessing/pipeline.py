"""
Preprocessing pipeline for ophthalmic images.

This module implements a configurable preprocessing pipeline with
steps optimized for ophthalmic image registration, including grayscale
conversion, contrast enhancement, vessel filtering, and resolution normalization.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Callable, Dict, Any
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from scipy import ndimage

from ophthalmic_registration.core.image_data import ImageData, PixelSpacing
from ophthalmic_registration.core.exceptions import PreprocessingError

logger = logging.getLogger(__name__)


class PreprocessingStep(Enum):
    """Available preprocessing steps."""
    GRAYSCALE = "grayscale"
    NORMALIZE_INTENSITY = "normalize_intensity"
    CLAHE = "clahe"
    GAUSSIAN_BLUR = "gaussian_blur"
    MEDIAN_FILTER = "median_filter"
    BILATERAL_FILTER = "bilateral_filter"
    VESSEL_ENHANCEMENT = "vessel_enhancement"
    UNSHARP_MASK = "unsharp_mask"
    RESIZE = "resize"
    CROP_CENTER = "crop_center"
    PAD_TO_SIZE = "pad_to_size"


@dataclass
class PreprocessingConfig:
    """
    Configuration for the preprocessing pipeline.
    
    Attributes:
        steps: Ordered list of preprocessing steps to apply
        target_size: Target image size (height, width) for resize operations
        clahe_clip_limit: CLAHE clip limit (typically 2.0-4.0)
        clahe_tile_size: CLAHE tile grid size
        gaussian_sigma: Gaussian blur sigma
        median_kernel_size: Median filter kernel size (must be odd)
        bilateral_d: Bilateral filter diameter
        bilateral_sigma_color: Bilateral filter color sigma
        bilateral_sigma_space: Bilateral filter spatial sigma
        vessel_sigma_range: Range of sigmas for vessel enhancement
        unsharp_amount: Unsharp mask amount
        unsharp_sigma: Unsharp mask sigma
        normalize_range: Target intensity range for normalization
        preserve_dtype: Preserve original data type after processing
    """
    steps: List[PreprocessingStep] = field(default_factory=lambda: [
        PreprocessingStep.GRAYSCALE,
        PreprocessingStep.CLAHE,
    ])
    
    # Resize parameters
    target_size: Optional[Tuple[int, int]] = None
    
    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    
    # Smoothing parameters
    gaussian_sigma: float = 1.0
    median_kernel_size: int = 3
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    
    # Vessel enhancement parameters
    vessel_sigma_range: Tuple[float, float, float] = (1.0, 3.0, 0.5)  # min, max, step
    
    # Unsharp mask parameters
    unsharp_amount: float = 1.0
    unsharp_sigma: float = 1.0
    
    # Normalization parameters
    normalize_range: Tuple[float, float] = (0.0, 255.0)
    preserve_dtype: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "steps": [s.value for s in self.steps],
            "target_size": self.target_size,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_size": self.clahe_tile_size,
            "gaussian_sigma": self.gaussian_sigma,
            "median_kernel_size": self.median_kernel_size,
            "bilateral_d": self.bilateral_d,
            "bilateral_sigma_color": self.bilateral_sigma_color,
            "bilateral_sigma_space": self.bilateral_sigma_space,
            "vessel_sigma_range": self.vessel_sigma_range,
            "unsharp_amount": self.unsharp_amount,
            "unsharp_sigma": self.unsharp_sigma,
            "normalize_range": self.normalize_range,
            "preserve_dtype": self.preserve_dtype,
        }


class PreprocessingPipeline:
    """
    Configurable preprocessing pipeline for ophthalmic images.
    
    Applies a sequence of preprocessing operations to prepare images
    for registration. Ensures consistent processing between baseline
    and follow-up images.
    
    Attributes:
        config: PreprocessingConfig defining the pipeline steps
    
    Example:
        >>> config = PreprocessingConfig(
        ...     steps=[
        ...         PreprocessingStep.GRAYSCALE,
        ...         PreprocessingStep.CLAHE,
        ...         PreprocessingStep.GAUSSIAN_BLUR,
        ...     ],
        ...     clahe_clip_limit=3.0,
        ...     gaussian_sigma=1.5
        ... )
        >>> pipeline = PreprocessingPipeline(config)
        >>> processed = pipeline.process(image_data)
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        
        Raises:
            ImportError: If OpenCV is not available
        """
        if not CV2_AVAILABLE:
            raise ImportError(
                "OpenCV is required for preprocessing. "
                "Install with: pip install opencv-python"
            )
        
        self.config = config or PreprocessingConfig()
        
        # Map steps to processing functions
        self._step_functions: Dict[PreprocessingStep, Callable] = {
            PreprocessingStep.GRAYSCALE: self._to_grayscale,
            PreprocessingStep.NORMALIZE_INTENSITY: self._normalize_intensity,
            PreprocessingStep.CLAHE: self._apply_clahe,
            PreprocessingStep.GAUSSIAN_BLUR: self._gaussian_blur,
            PreprocessingStep.MEDIAN_FILTER: self._median_filter,
            PreprocessingStep.BILATERAL_FILTER: self._bilateral_filter,
            PreprocessingStep.VESSEL_ENHANCEMENT: self._vessel_enhancement,
            PreprocessingStep.UNSHARP_MASK: self._unsharp_mask,
            PreprocessingStep.RESIZE: self._resize,
            PreprocessingStep.CROP_CENTER: self._crop_center,
            PreprocessingStep.PAD_TO_SIZE: self._pad_to_size,
        }
    
    def process(self, image_data: ImageData, in_place: bool = False) -> ImageData:
        """
        Apply the preprocessing pipeline to an image.
        
        Args:
            image_data: Input image data
            in_place: If True, modify the input; otherwise create a copy
        
        Returns:
            Preprocessed ImageData
        
        Raises:
            PreprocessingError: If any preprocessing step fails
        """
        if not in_place:
            image_data = image_data.copy()
        
        original_dtype = image_data.pixel_array.dtype
        
        for step in self.config.steps:
            try:
                step_func = self._step_functions.get(step)
                if step_func is None:
                    raise PreprocessingError(
                        step.value,
                        f"Unknown preprocessing step: {step.value}"
                    )
                
                image_data.pixel_array = step_func(image_data.pixel_array)
                image_data.preprocessing_history.append(step.value)
                
                logger.debug(f"Applied preprocessing step: {step.value}")
                
            except PreprocessingError:
                raise
            except Exception as e:
                raise PreprocessingError(
                    step.value,
                    f"Step failed with error: {e}",
                    details=str(type(e).__name__)
                )
        
        # Restore original dtype if requested
        if self.config.preserve_dtype and image_data.pixel_array.dtype != original_dtype:
            image_data.pixel_array = image_data.pixel_array.astype(original_dtype)
        
        image_data.is_preprocessed = True
        
        logger.info(
            f"Preprocessing complete: {len(self.config.steps)} steps applied, "
            f"output shape={image_data.shape}, dtype={image_data.dtype}"
        )
        
        return image_data
    
    def process_pair(
        self,
        baseline: ImageData,
        followup: ImageData,
        in_place: bool = False
    ) -> Tuple[ImageData, ImageData]:
        """
        Apply identical preprocessing to a pair of images.
        
        Ensures consistent preprocessing between baseline and follow-up
        images for accurate registration.
        
        Args:
            baseline: Baseline image data
            followup: Follow-up image data
            in_place: If True, modify inputs; otherwise create copies
        
        Returns:
            Tuple of (preprocessed_baseline, preprocessed_followup)
        """
        processed_baseline = self.process(baseline, in_place=in_place)
        processed_followup = self.process(followup, in_place=in_place)
        
        # Warn if sizes differ after preprocessing
        if processed_baseline.shape != processed_followup.shape:
            logger.warning(
                f"Image sizes differ after preprocessing: "
                f"baseline={processed_baseline.shape}, followup={processed_followup.shape}"
            )
        
        return processed_baseline, processed_followup
    
    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if img.ndim == 2:
            return img
        
        if img.ndim == 3:
            if img.shape[2] == 1:
                return img[:, :, 0]
            elif img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        
        raise PreprocessingError(
            "grayscale",
            f"Cannot convert image with shape {img.shape} to grayscale"
        )
    
    def _normalize_intensity(self, img: np.ndarray) -> np.ndarray:
        """Normalize intensity to target range."""
        target_min, target_max = self.config.normalize_range
        
        img_float = img.astype(np.float64)
        img_min, img_max = img_float.min(), img_float.max()
        
        if img_max > img_min:
            img_normalized = (img_float - img_min) / (img_max - img_min)
            img_normalized = img_normalized * (target_max - target_min) + target_min
        else:
            img_normalized = np.full_like(img_float, target_min)
        
        return img_normalized.astype(np.uint8) if target_max <= 255 else img_normalized
    
    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        # Ensure uint8 for CLAHE
        if img.dtype != np.uint8:
            img = self._normalize_to_uint8(img)
        
        # Handle color images
        if img.ndim == 3:
            # Convert to LAB and apply CLAHE to L channel
            if img.shape[2] == 3:
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(
                    clipLimit=self.config.clahe_clip_limit,
                    tileGridSize=self.config.clahe_tile_size
                )
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                raise PreprocessingError(
                    "clahe",
                    f"CLAHE not supported for {img.shape[2]}-channel images"
                )
        
        # Grayscale
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_size
        )
        return clahe.apply(img)
    
    def _gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur."""
        sigma = self.config.gaussian_sigma
        # Kernel size should be odd and large enough for the sigma
        ksize = int(np.ceil(sigma * 6)) | 1  # Ensure odd
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)
    
    def _median_filter(self, img: np.ndarray) -> np.ndarray:
        """Apply median filter."""
        ksize = self.config.median_kernel_size
        if ksize % 2 == 0:
            ksize += 1  # Must be odd
        return cv2.medianBlur(img, ksize)
    
    def _bilateral_filter(self, img: np.ndarray) -> np.ndarray:
        """Apply bilateral filter for edge-preserving smoothing."""
        return cv2.bilateralFilter(
            img,
            d=self.config.bilateral_d,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space
        )
    
    def _vessel_enhancement(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance vessel structures using multi-scale Frangi filtering.
        
        Uses the Hessian-based Frangi vesselness filter to enhance
        tubular structures (blood vessels) in the image.
        """
        if img.ndim != 2:
            raise PreprocessingError(
                "vessel_enhancement",
                "Vessel enhancement requires grayscale input"
            )
        
        img_float = img.astype(np.float64)
        if img_float.max() > 1:
            img_float = img_float / 255.0
        
        sigma_min, sigma_max, sigma_step = self.config.vessel_sigma_range
        sigmas = np.arange(sigma_min, sigma_max + sigma_step, sigma_step)
        
        vesselness = np.zeros_like(img_float)
        
        for sigma in sigmas:
            # Compute Hessian components
            Ixx = ndimage.gaussian_filter(img_float, sigma, order=(0, 2))
            Iyy = ndimage.gaussian_filter(img_float, sigma, order=(2, 0))
            Ixy = ndimage.gaussian_filter(img_float, sigma, order=(1, 1))
            
            # Eigenvalues of Hessian
            tmp = np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy ** 2)
            lambda1 = 0.5 * ((Ixx + Iyy) + tmp)
            lambda2 = 0.5 * ((Ixx + Iyy) - tmp)
            
            # Frangi vesselness
            with np.errstate(divide='ignore', invalid='ignore'):
                Rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)
                S = np.sqrt(lambda1 ** 2 + lambda2 ** 2)
                
                beta = 0.5
                c = 0.5 * S.max()
                
                vessel = np.exp(-Rb ** 2 / (2 * beta ** 2)) * (1 - np.exp(-S ** 2 / (2 * c ** 2)))
                vessel[lambda2 > 0] = 0  # Only dark vessels on light background
            
            vesselness = np.maximum(vesselness, vessel * sigma ** 2)
        
        # Normalize and convert back
        vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-10)
        return (vesselness * 255).astype(np.uint8)
    
    def _unsharp_mask(self, img: np.ndarray) -> np.ndarray:
        """Apply unsharp masking for edge enhancement."""
        sigma = self.config.unsharp_sigma
        amount = self.config.unsharp_amount
        
        # Create blurred version
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        
        # Unsharp mask: original + amount * (original - blurred)
        sharpened = cv2.addWeighted(
            img, 1.0 + amount,
            blurred, -amount,
            0
        )
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        if self.config.target_size is None:
            return img
        
        target_h, target_w = self.config.target_size
        current_h, current_w = img.shape[:2]
        
        if (current_h, current_w) == (target_h, target_w):
            return img
        
        # Use appropriate interpolation based on scaling direction
        if current_h * current_w > target_h * target_w:
            # Downscaling - use INTER_AREA
            interpolation = cv2.INTER_AREA
        else:
            # Upscaling - use INTER_CUBIC
            interpolation = cv2.INTER_CUBIC
        
        return cv2.resize(img, (target_w, target_h), interpolation=interpolation)
    
    def _crop_center(self, img: np.ndarray) -> np.ndarray:
        """Crop image to target size from center."""
        if self.config.target_size is None:
            return img
        
        target_h, target_w = self.config.target_size
        current_h, current_w = img.shape[:2]
        
        if current_h < target_h or current_w < target_w:
            raise PreprocessingError(
                "crop_center",
                f"Image ({current_h}x{current_w}) smaller than target ({target_h}x{target_w})"
            )
        
        start_y = (current_h - target_h) // 2
        start_x = (current_w - target_w) // 2
        
        if img.ndim == 2:
            return img[start_y:start_y + target_h, start_x:start_x + target_w]
        else:
            return img[start_y:start_y + target_h, start_x:start_x + target_w, :]
    
    def _pad_to_size(self, img: np.ndarray) -> np.ndarray:
        """Pad image to target size with zeros."""
        if self.config.target_size is None:
            return img
        
        target_h, target_w = self.config.target_size
        current_h, current_w = img.shape[:2]
        
        if current_h >= target_h and current_w >= target_w:
            return img
        
        pad_h = max(0, target_h - current_h)
        pad_w = max(0, target_w - current_w)
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        if img.ndim == 2:
            return np.pad(
                img,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0
            )
        else:
            return np.pad(
                img,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=0
            )
    
    def _normalize_to_uint8(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to uint8 range."""
        img_float = img.astype(np.float64)
        img_min, img_max = img_float.min(), img_float.max()
        
        if img_max > img_min:
            img_normalized = (img_float - img_min) / (img_max - img_min) * 255
        else:
            img_normalized = np.zeros_like(img_float)
        
        return img_normalized.astype(np.uint8)
    
    @staticmethod
    def create_default_fundus_pipeline() -> 'PreprocessingPipeline':
        """Create a preprocessing pipeline optimized for fundus images."""
        config = PreprocessingConfig(
            steps=[
                PreprocessingStep.GRAYSCALE,
                PreprocessingStep.CLAHE,
                PreprocessingStep.GAUSSIAN_BLUR,
            ],
            clahe_clip_limit=2.0,
            clahe_tile_size=(8, 8),
            gaussian_sigma=0.5,
        )
        return PreprocessingPipeline(config)
    
    @staticmethod
    def create_default_faf_pipeline() -> 'PreprocessingPipeline':
        """Create a preprocessing pipeline optimized for FAF images."""
        config = PreprocessingConfig(
            steps=[
                PreprocessingStep.GRAYSCALE,
                PreprocessingStep.NORMALIZE_INTENSITY,
                PreprocessingStep.CLAHE,
            ],
            clahe_clip_limit=3.0,
            clahe_tile_size=(16, 16),
        )
        return PreprocessingPipeline(config)
    
    @staticmethod
    def create_minimal_pipeline() -> 'PreprocessingPipeline':
        """Create a minimal preprocessing pipeline (grayscale only)."""
        config = PreprocessingConfig(
            steps=[PreprocessingStep.GRAYSCALE],
        )
        return PreprocessingPipeline(config)


def normalize_resolution(
    baseline: ImageData,
    followup: ImageData,
    target_spacing: Optional[float] = None
) -> Tuple[ImageData, ImageData]:
    """
    Normalize resolution between two images based on pixel spacing.
    
    Rescales images so they have matching pixel spacing, enabling
    accurate spatial measurements after registration.
    
    Args:
        baseline: Baseline image with pixel spacing metadata
        followup: Follow-up image with pixel spacing metadata
        target_spacing: Target pixel spacing in mm (uses baseline if None)
    
    Returns:
        Tuple of resolution-normalized (baseline, followup) images
    
    Raises:
        PreprocessingError: If pixel spacing is missing from both images
    """
    baseline_spacing = baseline.pixel_spacing
    followup_spacing = followup.pixel_spacing
    
    if baseline_spacing is None and followup_spacing is None:
        logger.warning(
            "No pixel spacing available for resolution normalization. "
            "Images will be used at original resolution."
        )
        return baseline, followup
    
    # Determine target spacing
    if target_spacing is None:
        if baseline_spacing:
            target_spacing = baseline_spacing.mean_spacing
        else:
            target_spacing = followup_spacing.mean_spacing
    
    def rescale_image(img_data: ImageData, current_spacing: Optional[PixelSpacing]) -> ImageData:
        if current_spacing is None:
            logger.warning(f"No spacing for {img_data.filepath}, using original resolution")
            return img_data
        
        scale_factor = current_spacing.mean_spacing / target_spacing
        
        if np.isclose(scale_factor, 1.0, rtol=0.01):
            return img_data
        
        new_h = int(img_data.height * scale_factor)
        new_w = int(img_data.width * scale_factor)
        
        if scale_factor > 1:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA
        
        result = img_data.copy()
        result.pixel_array = cv2.resize(
            img_data.pixel_array,
            (new_w, new_h),
            interpolation=interpolation
        )
        
        # Update pixel spacing
        result.metadata.pixel_spacing = PixelSpacing(
            row_spacing=target_spacing,
            column_spacing=target_spacing,
            unit=current_spacing.unit,
            source="normalized"
        )
        
        result.preprocessing_history.append(f"resolution_normalized_scale_{scale_factor:.3f}")
        
        logger.info(
            f"Rescaled image from {img_data.shape} to {result.shape} "
            f"(scale={scale_factor:.3f})"
        )
        
        return result
    
    normalized_baseline = rescale_image(baseline, baseline_spacing)
    normalized_followup = rescale_image(followup, followup_spacing)
    
    return normalized_baseline, normalized_followup
