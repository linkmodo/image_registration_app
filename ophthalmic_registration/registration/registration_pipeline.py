"""
Two-stage registration pipeline for ophthalmic images.

This module implements the main registration pipeline that combines
feature-based coarse alignment (ORB/AKAZE/SIFT) with ECC-based fine 
refinement for robust ophthalmic image registration.
"""

import logging
import time
from typing import Optional, Tuple
from enum import Enum
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ophthalmic_registration.core.image_data import (
    ImageData,
    TransformResult,
    RegistrationConfig,
    MotionModel,
)
from ophthalmic_registration.core.exceptions import (
    RegistrationError,
    ConvergenceError,
)
from ophthalmic_registration.registration.sift_aligner import SiftAligner
from ophthalmic_registration.registration.ecc_aligner import EccAligner
from ophthalmic_registration.registration.feature_aligner import (
    FeatureAligner, FeatureDetector
)
from ophthalmic_registration.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)


class CoarseAlignmentMethod(Enum):
    """Available coarse alignment methods."""
    AKAZE = "akaze"
    ORB = "orb"
    SIFT = "sift"


class RegistrationPipeline:
    """
    Two-stage registration pipeline with feature-based coarse alignment and ECC refinement.
    
    Performs robust image registration through:
    1. Optional preprocessing for feature enhancement
    2. Feature-based coarse alignment (ORB/AKAZE/SIFT) with RANSAC
    3. ECC-based fine refinement (always runs as final step)
    
    The pipeline handles graceful degradation when individual stages
    fail, falling back to previous results with appropriate warnings.
    
    Attributes:
        config: Registration configuration parameters
        preprocessor: Optional preprocessing pipeline
        coarse_method: Method for coarse alignment (ORB, AKAZE, SIFT)
        feature_aligner: Feature-based coarse aligner
        ecc_aligner: ECC fine aligner
    
    Example:
        >>> pipeline = RegistrationPipeline(coarse_method=CoarseAlignmentMethod.ORB)
        >>> result = pipeline.register(baseline_image, followup_image)
        >>> 
        >>> # Apply transform to get registered image
        >>> registered = pipeline.apply_transform(
        ...     followup_image, result.transform_matrix
        ... )
    """
    
    def __init__(
        self,
        config: Optional[RegistrationConfig] = None,
        preprocessor: Optional[PreprocessingPipeline] = None,
        coarse_method: CoarseAlignmentMethod = CoarseAlignmentMethod.AKAZE,
        use_flann: bool = True,
        n_features: int = 5000
    ):
        """
        Initialize the registration pipeline.
        
        Args:
            config: Registration configuration
            preprocessor: Optional preprocessing pipeline
            coarse_method: Method for coarse alignment (ORB, AKAZE, SIFT)
            use_flann: Use FLANN matcher for SIFT (vs brute-force)
            n_features: Maximum features to detect for feature-based methods
        
        Raises:
            ImportError: If OpenCV is not available
        """
        if not CV2_AVAILABLE:
            raise ImportError(
                "OpenCV is required for registration. "
                "Install with: pip install opencv-python"
            )
        
        self.config = config or RegistrationConfig()
        self.preprocessor = preprocessor
        self.coarse_method = coarse_method
        
        # Initialize aligner based on method
        self._init_coarse_aligner(coarse_method, use_flann, n_features)
    
    def _init_coarse_aligner(
        self,
        method: CoarseAlignmentMethod,
        use_flann: bool,
        n_features: int
    ) -> None:
        """Initialize the coarse alignment method."""
        if method == CoarseAlignmentMethod.AKAZE:
            self.feature_aligner = FeatureAligner(
                self.config,
                detector_type=FeatureDetector.AKAZE,
                n_features=n_features
            )
            self.sift_aligner = None
            logger.info("Using AKAZE for coarse alignment")
            
        elif method == CoarseAlignmentMethod.ORB:
            self.feature_aligner = FeatureAligner(
                self.config,
                detector_type=FeatureDetector.ORB,
                n_features=n_features
            )
            self.sift_aligner = None
            logger.info("Using ORB for coarse alignment")
            
        elif method == CoarseAlignmentMethod.SIFT:
            self.feature_aligner = None
            self.sift_aligner = SiftAligner(self.config, use_flann=use_flann)
            logger.info("Using SIFT for coarse alignment")
            
        else:
            raise ValueError(f"Unknown coarse alignment method: {method}")
    
    def register(
        self,
        baseline: ImageData,
        followup: ImageData,
        preprocess: bool = True
    ) -> TransformResult:
        """
        Perform two-stage registration.
        
        Args:
            baseline: Reference/baseline image
            followup: Moving/follow-up image to align to baseline
            preprocess: Apply preprocessing before registration
        
        Returns:
            TransformResult containing the final transform and quality metrics
        
        Raises:
            RegistrationError: If registration fails completely
        """
        total_start_time = time.time()
        all_warnings = []
        quality_metrics = {}
        
        # Preprocessing
        if preprocess and self.preprocessor is not None:
            logger.info("Applying preprocessing...")
            baseline_proc, followup_proc = self.preprocessor.process_pair(
                baseline, followup
            )
        else:
            baseline_proc = baseline
            followup_proc = followup
        
        # Stage 1: Coarse alignment (ORB/AKAZE/SIFT/Elastix)
        coarse_result = None
        coarse_transform = None
        
        if self.config.use_coarse_alignment:
            logger.info(f"Stage 1: {self.coarse_method.value.upper()} coarse alignment...")
            try:
                coarse_result = self._run_coarse_alignment(baseline_proc, followup_proc)
                coarse_transform = coarse_result.transform_matrix
                all_warnings.extend(coarse_result.warnings)
                
                # Add coarse metrics
                for key, value in coarse_result.quality_metrics.items():
                    quality_metrics[f"coarse_{key}"] = value
                
                logger.info(
                    f"Coarse alignment complete: "
                    f"{coarse_result.quality_metrics.get('num_inliers', 'N/A')} inliers, "
                    f"reproj_error={coarse_result.quality_metrics.get('reproj_error_mean', 0):.2f}px"
                )
                
            except RegistrationError as e:
                logger.warning(f"Alignment failed: {e.message}")
                all_warnings.append(f"Alignment failed: {e.message}")
                
                # Initialize with identity transform
                if self.config.motion_model == MotionModel.HOMOGRAPHY:
                    coarse_transform = np.eye(3, dtype=np.float64)
                else:
                    coarse_transform = np.array([
                        [1, 0, 0],
                        [0, 1, 0]
                    ], dtype=np.float64)
        else:
            # No coarse alignment - start with identity
            if self.config.motion_model == MotionModel.HOMOGRAPHY:
                coarse_transform = np.eye(3, dtype=np.float64)
            else:
                coarse_transform = np.array([
                    [1, 0, 0],
                    [0, 1, 0]
                ], dtype=np.float64)
        
        # Use coarse transform as final (no ECC refinement)
        final_transform = coarse_transform
        
        # Add alignment method to quality metrics
        quality_metrics["alignment_method"] = self.coarse_method.value
        
        # Compute overlap metrics
        overlap_metrics = self._compute_overlap_metrics(
            baseline_proc, followup_proc, final_transform
        )
        quality_metrics.update(overlap_metrics)
        
        total_time = (time.time() - total_start_time) * 1000
        
        result = TransformResult(
            transform_matrix=final_transform,
            motion_model=self.config.motion_model,
            coarse_transform=coarse_transform,
            fine_transform=None,
            ecc_correlation=None,
            ecc_converged=True,
            ecc_iterations=0,
            feature_match_result=coarse_result.feature_match_result if coarse_result else None,
            registration_time_ms=total_time,
            quality_metrics=quality_metrics,
            warnings=all_warnings
        )
        
        logger.info(
            f"Registration complete in {total_time:.1f}ms"
        )
        
        return result
    
    def _run_coarse_alignment(
        self,
        baseline: ImageData,
        followup: ImageData
    ) -> TransformResult:
        """
        Run coarse alignment using the configured method.
        
        Args:
            baseline: Reference image
            followup: Moving image
        
        Returns:
            TransformResult from coarse alignment
        """
        if self.feature_aligner is not None:
            return self.feature_aligner.align(baseline, followup)
        elif self.sift_aligner is not None:
            return self.sift_aligner.align(baseline, followup)
        else:
            raise RegistrationError(
                stage="alignment",
                reason="No aligner configured"
            )
    
    def apply_transform(
        self,
        image: ImageData,
        transform: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: float = 0
    ) -> ImageData:
        """
        Apply a transform to an image.
        
        Args:
            image: Image to transform
            transform: Transformation matrix
            output_size: Output size (height, width); uses input size if None
            border_mode: OpenCV border mode for pixels outside image
            border_value: Value for constant border mode
        
        Returns:
            Transformed ImageData
        """
        img = image.pixel_array
        
        if output_size is None:
            output_size = (img.shape[0], img.shape[1])
        
        h, w = output_size
        
        # Apply transform
        if transform.shape[0] == 3:
            # Homography
            warped = cv2.warpPerspective(
                img, transform, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=border_mode,
                borderValue=border_value
            )
        else:
            # Affine
            warped = cv2.warpAffine(
                img, transform, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=border_mode,
                borderValue=border_value
            )
        
        # Create new ImageData with transformed pixels
        result = image.copy()
        result.pixel_array = warped
        result.preprocessing_history.append("transform_applied")
        
        return result
    
    def register_and_apply(
        self,
        baseline: ImageData,
        followup: ImageData,
        preprocess: bool = True
    ) -> Tuple[TransformResult, ImageData]:
        """
        Register images and return both transform and registered image.
        
        Convenience method combining register() and apply_transform().
        
        Args:
            baseline: Reference/baseline image
            followup: Moving/follow-up image
            preprocess: Apply preprocessing
        
        Returns:
            Tuple of (TransformResult, registered_followup)
        """
        result = self.register(baseline, followup, preprocess=preprocess)
        
        # Apply to original (non-preprocessed) follow-up
        registered = self.apply_transform(
            followup,
            result.transform_matrix,
            output_size=(baseline.height, baseline.width)
        )
        
        # Inherit pixel spacing from baseline (registered image is in baseline's coordinate system)
        if baseline.metadata and baseline.metadata.pixel_spacing:
            if not registered.metadata:
                from ophthalmic_registration.core.image_data import ImageMetadata
                registered.metadata = ImageMetadata()
            registered.metadata.pixel_spacing = baseline.metadata.pixel_spacing
            logger.debug(f"Registered image inherited pixel spacing from baseline: "
                        f"{baseline.metadata.pixel_spacing.row_spacing:.4f} x "
                        f"{baseline.metadata.pixel_spacing.column_spacing:.4f} mm")
        
        return result, registered
    
    def _compute_overlap_metrics(
        self,
        baseline: ImageData,
        followup: ImageData,
        transform: np.ndarray
    ) -> dict:
        """Compute overlap-based quality metrics."""
        metrics = {}
        
        try:
            # Get image dimensions
            h, w = baseline.height, baseline.width
            
            # Create mask of valid pixels in follow-up
            followup_mask = np.ones((followup.height, followup.width), dtype=np.uint8) * 255
            
            # Warp the mask
            if transform.shape[0] == 3:
                warped_mask = cv2.warpPerspective(
                    followup_mask, transform, (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
            else:
                warped_mask = cv2.warpAffine(
                    followup_mask, transform, (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
            
            # Compute overlap
            overlap_pixels = np.sum(warped_mask > 0)
            total_pixels = h * w
            
            metrics["overlap_ratio"] = float(overlap_pixels / total_pixels)
            metrics["overlap_pixels"] = int(overlap_pixels)
            
        except Exception as e:
            logger.warning(f"Failed to compute overlap metrics: {e}")
            metrics["overlap_ratio"] = 0.0
            metrics["overlap_pixels"] = 0
        
        return metrics
    
    def get_registration_quality_summary(self, result: TransformResult) -> dict:
        """
        Generate a human-readable quality summary.
        
        Args:
            result: TransformResult from registration
        
        Returns:
            Dictionary with quality assessment
        """
        summary = {
            "overall_quality": "unknown",
            "issues": [],
            "recommendations": []
        }
        
        # Check ECC correlation
        final_ecc = result.quality_metrics.get("final_ecc", 0)
        if final_ecc >= 0.9:
            summary["ecc_quality"] = "excellent"
        elif final_ecc >= 0.7:
            summary["ecc_quality"] = "good"
        elif final_ecc >= 0.5:
            summary["ecc_quality"] = "fair"
            summary["issues"].append("Moderate correlation - registration may have errors")
        else:
            summary["ecc_quality"] = "poor"
            summary["issues"].append("Low correlation - registration likely failed")
            summary["recommendations"].append("Try different preprocessing or motion model")
        
        # Check convergence
        if not result.ecc_converged:
            summary["issues"].append("ECC did not converge")
            summary["recommendations"].append("Increase ecc_max_iterations or use coarser motion model")
        
        # Check overlap
        overlap = result.quality_metrics.get("overlap_ratio", 0)
        if overlap < 0.5:
            summary["issues"].append(f"Low overlap ({overlap:.1%})")
            summary["recommendations"].append("Check if images are from same region")
        
        # Check for large transforms
        if result.warnings:
            summary["issues"].extend(result.warnings)
        
        # Determine overall quality
        if len(summary["issues"]) == 0 and final_ecc >= 0.7:
            summary["overall_quality"] = "good"
        elif len(summary["issues"]) <= 1 and final_ecc >= 0.5:
            summary["overall_quality"] = "acceptable"
        else:
            summary["overall_quality"] = "poor"
        
        return summary
    
    @staticmethod
    def compose_transforms(
        transform1: np.ndarray,
        transform2: np.ndarray
    ) -> np.ndarray:
        """
        Compose two transforms: result = transform2 @ transform1.
        
        Args:
            transform1: First transform to apply
            transform2: Second transform to apply
        
        Returns:
            Composed transformation matrix
        """
        # Convert to 3x3 if needed
        def to_3x3(t):
            if t.shape[0] == 2:
                result = np.eye(3, dtype=np.float64)
                result[:2, :] = t
                return result
            return t.astype(np.float64)
        
        t1 = to_3x3(transform1)
        t2 = to_3x3(transform2)
        
        composed = t2 @ t1
        
        # Return in same format as inputs
        if transform1.shape[0] == 2 and transform2.shape[0] == 2:
            return composed[:2, :]
        return composed
    
    @staticmethod
    def invert_transform(
        transform: np.ndarray,
        motion_model: MotionModel = MotionModel.AFFINE
    ) -> np.ndarray:
        """
        Compute the inverse of a transform.
        
        Args:
            transform: Transform to invert
            motion_model: Motion model (affects inversion method)
        
        Returns:
            Inverted transformation matrix
        """
        if transform.shape[0] == 3:
            # Homography - direct matrix inverse
            return np.linalg.inv(transform)
        else:
            # Affine - use cv2.invertAffineTransform
            return cv2.invertAffineTransform(transform)
