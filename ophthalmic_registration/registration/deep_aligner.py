"""
Deep learning-based registration for multi-modality ophthalmic images.

This module implements registration using deep learning feature extractors
that are robust to appearance changes between different imaging modalities
(e.g., fundus to FAF, infrared to FAF).

Supported methods:
- SuperPoint + LightGlue: State-of-the-art learned features with attention-based matching
- LoFTR: Detector-free local feature matching with transformers

These methods excel at cross-modality registration where traditional
feature detectors (SIFT, ORB, AKAZE) often fail due to appearance differences.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any

import numpy as np

from ophthalmic_registration.core.image_data import ImageData, TransformResult, MotionModel

logger = logging.getLogger(__name__)

# Check for deep learning dependencies
TORCH_AVAILABLE = False
KORNIA_AVAILABLE = False
LIGHTGLUE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    logger.debug("PyTorch available")
except ImportError:
    logger.debug("PyTorch not available")

try:
    import kornia
    from kornia.feature import LoFTR
    KORNIA_AVAILABLE = True
    logger.debug("Kornia available")
except ImportError:
    logger.debug("Kornia not available")

try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd
    LIGHTGLUE_AVAILABLE = True
    logger.debug("LightGlue available")
except ImportError:
    logger.debug("LightGlue not available - install with: pip install git+https://github.com/cvg/LightGlue.git")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class DeepMatchingMethod(Enum):
    """Available deep learning matching methods."""
    SUPERPOINT_LIGHTGLUE = "superpoint_lightglue"
    LOFTR = "loftr"


@dataclass
class DeepAlignerConfig:
    """Configuration for deep learning aligner."""
    method: DeepMatchingMethod = DeepMatchingMethod.SUPERPOINT_LIGHTGLUE
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    max_keypoints: int = 2048
    match_threshold: float = 0.2
    min_matches: int = 10
    ransac_threshold: float = 5.0
    ransac_confidence: float = 0.999
    ransac_max_iters: int = 2000


def is_deep_learning_available() -> bool:
    """Check if deep learning registration is available."""
    return TORCH_AVAILABLE and (LIGHTGLUE_AVAILABLE or KORNIA_AVAILABLE)


def get_available_methods() -> list:
    """Get list of available deep learning methods."""
    methods = []
    if TORCH_AVAILABLE:
        if LIGHTGLUE_AVAILABLE:
            methods.append(DeepMatchingMethod.SUPERPOINT_LIGHTGLUE)
        if KORNIA_AVAILABLE:
            methods.append(DeepMatchingMethod.LOFTR)
    return methods


class DeepAligner:
    """
    Deep learning-based image aligner for multi-modality registration.
    
    Uses learned feature extractors and matchers that are robust to
    appearance changes between different imaging modalities.
    
    Attributes:
        config: Aligner configuration
        device: PyTorch device (cuda/cpu)
        extractor: Feature extractor model
        matcher: Feature matcher model
    """
    
    def __init__(self, config: Optional[DeepAlignerConfig] = None):
        """
        Initialize the deep aligner.
        
        Args:
            config: Aligner configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for deep learning registration. "
                "Install with: pip install torch torchvision"
            )
        
        self.config = config or DeepAlignerConfig()
        self.device = torch.device(self.config.device)
        
        self._extractor = None
        self._matcher = None
        self._loftr = None
        
        self._init_models()
    
    def _init_models(self) -> None:
        """Initialize the deep learning models."""
        if self.config.method == DeepMatchingMethod.SUPERPOINT_LIGHTGLUE:
            if not LIGHTGLUE_AVAILABLE:
                raise ImportError(
                    "LightGlue is required for SuperPoint+LightGlue matching. "
                    "Install with: pip install git+https://github.com/cvg/LightGlue.git"
                )
            
            logger.info(f"Loading SuperPoint + LightGlue on {self.device}")
            self._extractor = SuperPoint(max_num_keypoints=self.config.max_keypoints).eval().to(self.device)
            self._matcher = LightGlue(features='superpoint').eval().to(self.device)
            logger.info("SuperPoint + LightGlue loaded successfully")
            
        elif self.config.method == DeepMatchingMethod.LOFTR:
            if not KORNIA_AVAILABLE:
                raise ImportError(
                    "Kornia is required for LoFTR matching. "
                    "Install with: pip install kornia"
                )
            
            logger.info(f"Loading LoFTR on {self.device}")
            self._loftr = LoFTR(pretrained='outdoor').eval().to(self.device)
            logger.info("LoFTR loaded successfully")
    
    def _preprocess_image(self, image: ImageData) -> torch.Tensor:
        """
        Preprocess image for deep learning models.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed tensor [1, 1, H, W] for grayscale or [1, 3, H, W] for RGB
        """
        img = image.as_float32()
        
        # Convert to grayscale if needed
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Normalize to [0, 1]
        if img.max() > 1.0:
            img = img / 255.0
        
        # Convert to tensor [1, 1, H, W]
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def _match_superpoint_lightglue(
        self,
        baseline: ImageData,
        followup: ImageData
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match images using SuperPoint + LightGlue.
        
        Returns:
            Tuple of (baseline_points, followup_points, confidence_scores)
        """
        # Preprocess images
        img0 = self._preprocess_image(baseline)
        img1 = self._preprocess_image(followup)
        
        with torch.no_grad():
            # Extract features
            feats0 = self._extractor.extract(img0)
            feats1 = self._extractor.extract(img1)
            
            # Match features
            matches01 = self._matcher({'image0': feats0, 'image1': feats1})
            
            # Get matched keypoints
            kpts0 = feats0['keypoints'][0].cpu().numpy()
            kpts1 = feats1['keypoints'][0].cpu().numpy()
            matches = matches01['matches'][0].cpu().numpy()
            scores = matches01['scores'][0].cpu().numpy()
            
            # Filter valid matches
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mscores = scores[valid]
        
        logger.info(f"SuperPoint+LightGlue: {len(mkpts0)} matches found")
        return mkpts0, mkpts1, mscores
    
    def _match_loftr(
        self,
        baseline: ImageData,
        followup: ImageData
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match images using LoFTR.
        
        Returns:
            Tuple of (baseline_points, followup_points, confidence_scores)
        """
        # Preprocess images
        img0 = self._preprocess_image(baseline)
        img1 = self._preprocess_image(followup)
        
        with torch.no_grad():
            # LoFTR expects dict input
            input_dict = {
                'image0': img0,
                'image1': img1
            }
            
            # Run LoFTR
            correspondences = self._loftr(input_dict)
            
            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()
            mconf = correspondences['confidence'].cpu().numpy()
        
        # Filter by confidence
        mask = mconf > self.config.match_threshold
        mkpts0 = mkpts0[mask]
        mkpts1 = mkpts1[mask]
        mconf = mconf[mask]
        
        logger.info(f"LoFTR: {len(mkpts0)} matches found (threshold={self.config.match_threshold})")
        return mkpts0, mkpts1, mconf
    
    def _estimate_transform(
        self,
        pts_baseline: np.ndarray,
        pts_followup: np.ndarray,
        motion_model: MotionModel = MotionModel.AFFINE
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Estimate transform from matched points using RANSAC.
        
        Args:
            pts_baseline: Points in baseline image [N, 2]
            pts_followup: Points in followup image [N, 2]
            motion_model: Type of transform to estimate
            
        Returns:
            Tuple of (transform_matrix, inlier_mask, num_inliers)
        """
        if len(pts_baseline) < self.config.min_matches:
            raise ValueError(
                f"Insufficient matches: {len(pts_baseline)} found, "
                f"{self.config.min_matches} required"
            )
        
        pts_baseline = pts_baseline.astype(np.float32)
        pts_followup = pts_followup.astype(np.float32)
        
        if motion_model == MotionModel.HOMOGRAPHY:
            transform, mask = cv2.findHomography(
                pts_followup, pts_baseline,
                cv2.RANSAC,
                self.config.ransac_threshold,
                maxIters=self.config.ransac_max_iters,
                confidence=self.config.ransac_confidence
            )
        elif motion_model == MotionModel.AFFINE:
            transform, mask = cv2.estimateAffine2D(
                pts_followup, pts_baseline,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_threshold,
                maxIters=self.config.ransac_max_iters,
                confidence=self.config.ransac_confidence
            )
        elif motion_model == MotionModel.EUCLIDEAN:
            transform, mask = cv2.estimateAffinePartial2D(
                pts_followup, pts_baseline,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_threshold,
                maxIters=self.config.ransac_max_iters,
                confidence=self.config.ransac_confidence
            )
        else:  # TRANSLATION
            # For translation, just compute mean shift
            shift = np.mean(pts_baseline - pts_followup, axis=0)
            transform = np.array([
                [1, 0, shift[0]],
                [0, 1, shift[1]]
            ], dtype=np.float64)
            mask = np.ones(len(pts_baseline), dtype=np.uint8)
        
        if transform is None:
            raise ValueError("Transform estimation failed")
        
        num_inliers = int(mask.sum()) if mask is not None else len(pts_baseline)
        return transform, mask, num_inliers
    
    def align(
        self,
        baseline: ImageData,
        followup: ImageData,
        motion_model: MotionModel = MotionModel.AFFINE
    ) -> TransformResult:
        """
        Align followup image to baseline using deep learning features.
        
        Args:
            baseline: Reference image
            followup: Moving image to align
            motion_model: Type of transform to estimate
            
        Returns:
            TransformResult with estimated transform
        """
        import time
        start_time = time.time()
        
        # Match features based on method
        if self.config.method == DeepMatchingMethod.SUPERPOINT_LIGHTGLUE:
            pts_baseline, pts_followup, scores = self._match_superpoint_lightglue(
                baseline, followup
            )
        else:  # LOFTR
            pts_baseline, pts_followup, scores = self._match_loftr(
                baseline, followup
            )
        
        # Estimate transform
        transform, mask, num_inliers = self._estimate_transform(
            pts_baseline, pts_followup, motion_model
        )
        
        # Compute reprojection error for inliers
        if mask is not None:
            inlier_pts_followup = pts_followup[mask.ravel() == 1]
            inlier_pts_baseline = pts_baseline[mask.ravel() == 1]
            
            if motion_model == MotionModel.HOMOGRAPHY:
                pts_h = np.hstack([inlier_pts_followup, np.ones((len(inlier_pts_followup), 1))])
                projected = (transform @ pts_h.T).T
                projected = projected[:, :2] / projected[:, 2:3]
            else:
                pts_h = np.hstack([inlier_pts_followup, np.ones((len(inlier_pts_followup), 1))])
                projected = (transform @ pts_h.T).T
            
            errors = np.linalg.norm(projected - inlier_pts_baseline, axis=1)
            reproj_error_mean = float(np.mean(errors))
            reproj_error_std = float(np.std(errors))
        else:
            reproj_error_mean = 0.0
            reproj_error_std = 0.0
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Build quality metrics
        quality_metrics = {
            'num_matches': len(pts_baseline),
            'num_inliers': num_inliers,
            'inlier_ratio': num_inliers / len(pts_baseline) if len(pts_baseline) > 0 else 0,
            'reproj_error_mean': reproj_error_mean,
            'reproj_error_std': reproj_error_std,
            'method': self.config.method.value,
            'match_confidence_mean': float(np.mean(scores)) if len(scores) > 0 else 0,
        }
        
        # Create feature match result for compatibility
        from ophthalmic_registration.core.image_data import FeatureMatchResult
        feature_result = FeatureMatchResult(
            num_keypoints_baseline=len(pts_baseline),
            num_keypoints_followup=len(pts_followup),
            num_matches=len(pts_baseline),
            num_inliers=num_inliers,
            inlier_ratio=quality_metrics['inlier_ratio'],
            reproj_error_mean=reproj_error_mean,
            reproj_error_std=reproj_error_std
        )
        
        return TransformResult(
            transform_matrix=transform,
            motion_model=motion_model,
            coarse_transform=transform,
            fine_transform=None,
            ecc_correlation=None,
            ecc_converged=True,
            ecc_iterations=0,
            feature_match_result=feature_result,
            registration_time_ms=elapsed_ms,
            quality_metrics=quality_metrics,
            warnings=[]
        )
