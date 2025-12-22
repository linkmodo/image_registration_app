"""
ECC-based fine alignment for ophthalmic images.

This module implements intensity-based image registration using the
Enhanced Correlation Coefficient (ECC) algorithm for sub-pixel accurate
refinement of an initial transform estimate.
"""

import logging
import time
from typing import Optional, Tuple
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

logger = logging.getLogger(__name__)


class EccAligner:
    """
    ECC-based fine image registration.
    
    Uses OpenCV's findTransformECC to refine an initial transform
    estimate through iterative intensity-based optimization.
    
    The ECC algorithm maximizes the Enhanced Correlation Coefficient
    between the template (baseline) and warped input (follow-up) images.
    
    Attributes:
        config: Registration configuration parameters
    
    Example:
        >>> config = RegistrationConfig(
        ...     motion_model=MotionModel.AFFINE,
        ...     ecc_max_iterations=500,
        ...     ecc_epsilon=1e-5
        ... )
        >>> aligner = EccAligner(config)
        >>> result = aligner.refine(
        ...     baseline_image,
        ...     followup_image,
        ...     initial_transform
        ... )
    """
    
    def __init__(self, config: Optional[RegistrationConfig] = None):
        """
        Initialize the ECC aligner.
        
        Args:
            config: Registration configuration
        
        Raises:
            ImportError: If OpenCV is not available
        """
        if not CV2_AVAILABLE:
            raise ImportError(
                "OpenCV is required for ECC alignment. "
                "Install with: pip install opencv-python"
            )
        
        self.config = config or RegistrationConfig()
    
    def refine(
        self,
        baseline: ImageData,
        followup: ImageData,
        initial_transform: np.ndarray,
        motion_model: Optional[MotionModel] = None
    ) -> TransformResult:
        """
        Refine transform using ECC optimization.
        
        Args:
            baseline: Reference/baseline image (template)
            followup: Moving/follow-up image to align
            initial_transform: Initial transform estimate (from coarse alignment)
            motion_model: Motion model to use (overrides config if provided)
        
        Returns:
            TransformResult with refined transformation
        
        Raises:
            ConvergenceError: If ECC fails to converge
            RegistrationError: If ECC optimization fails
        """
        start_time = time.time()
        warnings = []
        
        motion_model = motion_model or self.config.motion_model
        
        # Prepare images
        img_baseline = self._prepare_image(baseline)
        img_followup = self._prepare_image(followup)
        
        # Ensure images are same size
        if img_baseline.shape != img_followup.shape:
            logger.warning(
                f"Image size mismatch: baseline={img_baseline.shape}, "
                f"followup={img_followup.shape}. Resizing follow-up to match."
            )
            img_followup = cv2.resize(
                img_followup,
                (img_baseline.shape[1], img_baseline.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            warnings.append("Follow-up image resized to match baseline dimensions")
        
        # Prepare initial transform for ECC
        warp_matrix = self._prepare_warp_matrix(initial_transform, motion_model)
        
        # Define ECC termination criteria
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.config.ecc_max_iterations,
            self.config.ecc_epsilon
        )
        
        # Run ECC optimization
        try:
            cc, refined_warp = cv2.findTransformECC(
                templateImage=img_baseline,
                inputImage=img_followup,
                warpMatrix=warp_matrix,
                motionType=motion_model.opencv_flag,
                criteria=criteria,
                inputMask=None,
                gaussFiltSize=self.config.ecc_gauss_filt_size
            )
            converged = True
            
            logger.info(
                f"ECC converged with correlation coefficient: {cc:.6f}"
            )
            
        except cv2.error as e:
            error_msg = str(e)
            
            # Check for common non-convergence cases
            if "ecc" in error_msg.lower() or "converge" in error_msg.lower():
                logger.warning(f"ECC did not converge: {error_msg}")
                
                # Return initial transform with warning
                refined_warp = warp_matrix.copy()
                cc = self._compute_correlation(img_baseline, img_followup, refined_warp, motion_model)
                converged = False
                warnings.append(
                    f"ECC did not converge. Using initial transform. "
                    f"Correlation: {cc:.4f}"
                )
            else:
                raise RegistrationError(
                    stage="ecc_refinement",
                    reason=f"ECC optimization failed: {error_msg}"
                )
        
        # Convert back to standard transform format
        final_transform = self._convert_warp_to_transform(refined_warp, motion_model)
        
        # Compute quality metrics
        quality_metrics = {
            "ecc_correlation": float(cc),
            "ecc_converged": converged,
        }
        
        # Add transform change metrics
        transform_change = self._compute_transform_change(
            initial_transform, final_transform, motion_model
        )
        quality_metrics.update(transform_change)
        
        elapsed_time = (time.time() - start_time) * 1000
        
        # Estimate iterations (not directly available from OpenCV)
        # Use correlation as proxy for convergence quality
        estimated_iterations = self.config.ecc_max_iterations if not converged else -1
        
        return TransformResult(
            transform_matrix=final_transform,
            motion_model=motion_model,
            fine_transform=final_transform.copy(),
            ecc_correlation=float(cc),
            ecc_converged=converged,
            ecc_iterations=estimated_iterations,
            registration_time_ms=elapsed_time,
            quality_metrics=quality_metrics,
            warnings=warnings
        )
    
    def align_without_initial(
        self,
        baseline: ImageData,
        followup: ImageData,
        motion_model: Optional[MotionModel] = None
    ) -> TransformResult:
        """
        Perform ECC alignment starting from identity transform.
        
        Use this when no coarse alignment is available. Note that ECC
        may fail to converge if images are significantly misaligned.
        
        Args:
            baseline: Reference/baseline image
            followup: Moving/follow-up image
            motion_model: Motion model to use
        
        Returns:
            TransformResult with computed transformation
        """
        motion_model = motion_model or self.config.motion_model
        
        # Create identity transform
        if motion_model == MotionModel.HOMOGRAPHY:
            initial_transform = np.eye(3, dtype=np.float32)
        else:
            initial_transform = np.array([
                [1, 0, 0],
                [0, 1, 0]
            ], dtype=np.float32)
        
        return self.refine(baseline, followup, initial_transform, motion_model)
    
    def _prepare_image(self, image_data: ImageData) -> np.ndarray:
        """Prepare image for ECC optimization."""
        img = image_data.pixel_array
        
        # Convert to grayscale if needed
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # ECC works best with float32 images
        img = img.astype(np.float32)
        
        # Normalize to [0, 1] range
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        
        return img
    
    def _prepare_warp_matrix(
        self,
        transform: np.ndarray,
        motion_model: MotionModel
    ) -> np.ndarray:
        """
        Prepare warp matrix for ECC in correct format.
        
        ECC expects specific matrix shapes for each motion type.
        """
        transform = transform.astype(np.float32)
        
        if motion_model == MotionModel.HOMOGRAPHY:
            # 3x3 homography
            if transform.shape == (3, 3):
                return transform.copy()
            elif transform.shape == (2, 3):
                # Convert affine to homography
                warp = np.zeros((3, 3), dtype=np.float32)
                warp[:2, :] = transform
                warp[2, 2] = 1.0
                return warp
            else:
                raise RegistrationError(
                    stage="ecc_preparation",
                    reason=f"Invalid transform shape for homography: {transform.shape}"
                )
        
        elif motion_model == MotionModel.TRANSLATION:
            # 2x3 with only translation
            if transform.shape == (2, 3):
                warp = np.array([
                    [1, 0, transform[0, 2]],
                    [0, 1, transform[1, 2]]
                ], dtype=np.float32)
                return warp
            elif transform.shape == (3, 3):
                warp = np.array([
                    [1, 0, transform[0, 2]],
                    [0, 1, transform[1, 2]]
                ], dtype=np.float32)
                return warp
            else:
                raise RegistrationError(
                    stage="ecc_preparation",
                    reason=f"Invalid transform shape: {transform.shape}"
                )
        
        elif motion_model == MotionModel.EUCLIDEAN:
            # 2x3 with rotation and translation
            if transform.shape == (2, 3):
                return transform.copy()
            elif transform.shape == (3, 3):
                return transform[:2, :].copy()
            else:
                raise RegistrationError(
                    stage="ecc_preparation",
                    reason=f"Invalid transform shape: {transform.shape}"
                )
        
        elif motion_model == MotionModel.AFFINE:
            # 2x3 affine
            if transform.shape == (2, 3):
                return transform.copy()
            elif transform.shape == (3, 3):
                return transform[:2, :].copy()
            else:
                raise RegistrationError(
                    stage="ecc_preparation",
                    reason=f"Invalid transform shape: {transform.shape}"
                )
        
        else:
            raise RegistrationError(
                stage="ecc_preparation",
                reason=f"Unsupported motion model: {motion_model}"
            )
    
    def _convert_warp_to_transform(
        self,
        warp: np.ndarray,
        motion_model: MotionModel
    ) -> np.ndarray:
        """Convert ECC warp matrix to standard transform format."""
        if motion_model == MotionModel.HOMOGRAPHY:
            return warp.astype(np.float64)
        else:
            # All other models return 2x3
            if warp.shape[0] == 3:
                return warp[:2, :].astype(np.float64)
            return warp.astype(np.float64)
    
    def _compute_correlation(
        self,
        img_baseline: np.ndarray,
        img_followup: np.ndarray,
        warp: np.ndarray,
        motion_model: MotionModel
    ) -> float:
        """
        Compute correlation coefficient between baseline and warped follow-up.
        
        Used when ECC fails to converge to still report a quality metric.
        """
        try:
            # Warp the follow-up image
            if motion_model == MotionModel.HOMOGRAPHY:
                warped = cv2.warpPerspective(
                    img_followup, warp,
                    (img_baseline.shape[1], img_baseline.shape[0]),
                    flags=cv2.INTER_LINEAR
                )
            else:
                warped = cv2.warpAffine(
                    img_followup, warp,
                    (img_baseline.shape[1], img_baseline.shape[0]),
                    flags=cv2.INTER_LINEAR
                )
            
            # Create mask for valid pixels
            mask = (warped > 0).astype(np.float32)
            
            # Compute normalized correlation
            baseline_masked = img_baseline * mask
            warped_masked = warped * mask
            
            baseline_mean = baseline_masked.sum() / (mask.sum() + 1e-10)
            warped_mean = warped_masked.sum() / (mask.sum() + 1e-10)
            
            baseline_centered = (baseline_masked - baseline_mean * mask) * mask
            warped_centered = (warped_masked - warped_mean * mask) * mask
            
            numerator = (baseline_centered * warped_centered).sum()
            denominator = np.sqrt(
                (baseline_centered ** 2).sum() * (warped_centered ** 2).sum()
            ) + 1e-10
            
            return float(numerator / denominator)
            
        except Exception as e:
            logger.warning(f"Failed to compute correlation: {e}")
            return 0.0
    
    def _compute_transform_change(
        self,
        initial: np.ndarray,
        final: np.ndarray,
        motion_model: MotionModel
    ) -> dict:
        """Compute metrics describing how much the transform changed."""
        metrics = {}
        
        # Translation change
        initial_tx = initial[0, 2]
        initial_ty = initial[1, 2]
        final_tx = final[0, 2]
        final_ty = final[1, 2]
        
        metrics["translation_change_x"] = float(final_tx - initial_tx)
        metrics["translation_change_y"] = float(final_ty - initial_ty)
        metrics["translation_change_magnitude"] = float(
            np.sqrt((final_tx - initial_tx) ** 2 + (final_ty - initial_ty) ** 2)
        )
        
        # Rotation change (for non-translation models)
        if motion_model != MotionModel.TRANSLATION:
            initial_angle = np.degrees(np.arctan2(initial[1, 0], initial[0, 0]))
            final_angle = np.degrees(np.arctan2(final[1, 0], final[0, 0]))
            metrics["rotation_change_degrees"] = float(final_angle - initial_angle)
        
        return metrics
    
    def compute_ecc_score(
        self,
        baseline: ImageData,
        followup: ImageData,
        transform: np.ndarray,
        motion_model: Optional[MotionModel] = None
    ) -> float:
        """
        Compute ECC score for a given transform without optimization.
        
        Useful for evaluating registration quality or comparing transforms.
        
        Args:
            baseline: Reference image
            followup: Moving image
            transform: Transform to evaluate
            motion_model: Motion model (for warp type)
        
        Returns:
            ECC correlation coefficient
        """
        motion_model = motion_model or self.config.motion_model
        
        img_baseline = self._prepare_image(baseline)
        img_followup = self._prepare_image(followup)
        
        warp = self._prepare_warp_matrix(transform, motion_model)
        
        return self._compute_correlation(img_baseline, img_followup, warp, motion_model)
