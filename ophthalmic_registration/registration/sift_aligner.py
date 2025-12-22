"""
SIFT-based coarse alignment for ophthalmic images.

This module implements feature-based image registration using SIFT
(Scale-Invariant Feature Transform) with FLANN or brute-force matching
and RANSAC-based outlier rejection.
"""

import logging
import time
from typing import Optional, Tuple, List
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ophthalmic_registration.core.image_data import (
    ImageData,
    TransformResult,
    FeatureMatchResult,
    RegistrationConfig,
    MotionModel,
)
from ophthalmic_registration.core.exceptions import (
    RegistrationError,
    InsufficientFeaturesError,
    MatchingError,
    TransformValidationError,
)

logger = logging.getLogger(__name__)


class SiftAligner:
    """
    SIFT-based coarse image registration.
    
    Performs feature detection, matching, and geometric transform estimation
    using SIFT features with RANSAC-based robust estimation.
    
    Attributes:
        config: Registration configuration parameters
        sift: OpenCV SIFT detector instance
        matcher: Feature matcher (FLANN or BFMatcher)
    
    Example:
        >>> config = RegistrationConfig(
        ...     motion_model=MotionModel.AFFINE,
        ...     sift_n_features=3000,
        ...     match_ratio_threshold=0.7
        ... )
        >>> aligner = SiftAligner(config)
        >>> result = aligner.align(baseline_image, followup_image)
    """
    
    def __init__(
        self,
        config: Optional[RegistrationConfig] = None,
        use_flann: bool = True
    ):
        """
        Initialize the SIFT aligner.
        
        Args:
            config: Registration configuration
            use_flann: Use FLANN matcher (faster) vs brute-force
        
        Raises:
            ImportError: If OpenCV is not available
        """
        if not CV2_AVAILABLE:
            raise ImportError(
                "OpenCV is required for SIFT alignment. "
                "Install with: pip install opencv-contrib-python"
            )
        
        self.config = config or RegistrationConfig()
        self.use_flann = use_flann
        
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create(
            nfeatures=self.config.sift_n_features,
            nOctaveLayers=self.config.sift_n_octave_layers,
            contrastThreshold=self.config.sift_contrast_threshold,
            edgeThreshold=self.config.sift_edge_threshold
        )
        
        # Initialize matcher
        if use_flann:
            # FLANN parameters for SIFT
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def align(
        self,
        baseline: ImageData,
        followup: ImageData,
        initial_transform: Optional[np.ndarray] = None
    ) -> TransformResult:
        """
        Perform SIFT-based coarse alignment.
        
        Args:
            baseline: Reference/baseline image
            followup: Moving/follow-up image to align
            initial_transform: Optional initial transform estimate
        
        Returns:
            TransformResult with computed transformation
        
        Raises:
            InsufficientFeaturesError: If too few features detected
            MatchingError: If too few matches found
            RegistrationError: If transform estimation fails
        """
        start_time = time.time()
        warnings = []
        
        # Prepare images for feature detection
        img_baseline = self._prepare_image(baseline)
        img_followup = self._prepare_image(followup)
        
        # Detect and compute features
        kp_baseline, desc_baseline = self._detect_features(
            img_baseline, "baseline"
        )
        kp_followup, desc_followup = self._detect_features(
            img_followup, "followup"
        )
        
        logger.info(
            f"Features detected - baseline: {len(kp_baseline)}, "
            f"followup: {len(kp_followup)}"
        )
        
        # Match features
        matches = self._match_features(desc_baseline, desc_followup)
        
        if len(matches) < self.config.min_matches_required:
            raise MatchingError(
                matches_found=len(matches),
                required=self.config.min_matches_required,
                details="Try adjusting match_ratio_threshold or increasing feature count"
            )
        
        logger.info(f"Feature matches after ratio test: {len(matches)}")
        
        # Extract matched point coordinates
        pts_baseline = np.float32([
            kp_baseline[m.queryIdx].pt for m in matches
        ]).reshape(-1, 1, 2)
        pts_followup = np.float32([
            kp_followup[m.trainIdx].pt for m in matches
        ]).reshape(-1, 1, 2)
        
        # Estimate transform with RANSAC
        transform_matrix, inlier_mask = self._estimate_transform(
            pts_baseline, pts_followup
        )
        
        # Count inliers
        num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        
        if num_inliers < self.config.min_matches_required:
            raise MatchingError(
                matches_found=num_inliers,
                required=self.config.min_matches_required,
                details=f"Only {num_inliers} inliers after RANSAC"
            )
        
        logger.info(f"RANSAC inliers: {num_inliers}/{len(matches)}")
        
        # Validate transform if configured
        if self.config.validate_transform:
            validation_warnings = self._validate_transform(transform_matrix)
            warnings.extend(validation_warnings)
        
        # Create feature match result
        feature_result = FeatureMatchResult(
            keypoints_baseline=kp_baseline,
            keypoints_followup=kp_followup,
            descriptors_baseline=desc_baseline,
            descriptors_followup=desc_followup,
            matches=matches,
            inlier_mask=inlier_mask,
            num_inliers=num_inliers
        )
        
        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(
            pts_baseline, pts_followup, transform_matrix, inlier_mask
        )
        
        elapsed_time = (time.time() - start_time) * 1000
        
        return TransformResult(
            transform_matrix=transform_matrix,
            motion_model=self.config.motion_model,
            coarse_transform=transform_matrix.copy(),
            feature_match_result=feature_result,
            registration_time_ms=elapsed_time,
            quality_metrics=quality_metrics,
            warnings=warnings
        )
    
    def _prepare_image(self, image_data: ImageData) -> np.ndarray:
        """Prepare image for SIFT feature detection."""
        img = image_data.pixel_array
        
        # Convert to grayscale if needed
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Ensure uint8
        if img.dtype != np.uint8:
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        
        return img
    
    def _detect_features(
        self,
        img: np.ndarray,
        image_name: str
    ) -> Tuple[List, np.ndarray]:
        """
        Detect SIFT keypoints and compute descriptors.
        
        Args:
            img: Grayscale image
            image_name: Name for logging/errors
        
        Returns:
            Tuple of (keypoints, descriptors)
        
        Raises:
            InsufficientFeaturesError: If too few features detected
        """
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        
        if len(keypoints) < self.config.min_matches_required:
            raise InsufficientFeaturesError(
                detected=len(keypoints),
                required=self.config.min_matches_required,
                image_name=image_name,
                details="Try reducing contrast_threshold or increasing n_features"
            )
        
        return keypoints, descriptors
    
    def _match_features(
        self,
        desc_baseline: np.ndarray,
        desc_followup: np.ndarray
    ) -> List:
        """
        Match features using kNN and Lowe's ratio test.
        
        Args:
            desc_baseline: Descriptors from baseline image
            desc_followup: Descriptors from follow-up image
        
        Returns:
            List of good matches after ratio test
        """
        # kNN matching with k=2 for ratio test
        try:
            raw_matches = self.matcher.knnMatch(desc_baseline, desc_followup, k=2)
        except cv2.error as e:
            raise RegistrationError(
                stage="feature_matching",
                reason=f"OpenCV matcher failed: {e}"
            )
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.config.match_ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def _estimate_transform(
        self,
        pts_baseline: np.ndarray,
        pts_followup: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate geometric transform using RANSAC.
        
        Args:
            pts_baseline: Matched points in baseline image (Nx1x2)
            pts_followup: Matched points in follow-up image (Nx1x2)
        
        Returns:
            Tuple of (transform_matrix, inlier_mask)
        
        Raises:
            RegistrationError: If transform estimation fails
        """
        motion_model = self.config.motion_model
        
        try:
            if motion_model == MotionModel.HOMOGRAPHY:
                transform, mask = cv2.findHomography(
                    pts_followup, pts_baseline,  # src -> dst
                    cv2.RANSAC,
                    ransacReprojThreshold=self.config.ransac_reproj_threshold,
                    maxIters=self.config.ransac_max_iters,
                    confidence=self.config.ransac_confidence
                )
            elif motion_model == MotionModel.AFFINE:
                transform, mask = cv2.estimateAffine2D(
                    pts_followup, pts_baseline,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.config.ransac_reproj_threshold,
                    maxIters=self.config.ransac_max_iters,
                    confidence=self.config.ransac_confidence
                )
            elif motion_model == MotionModel.EUCLIDEAN:
                transform, mask = cv2.estimateAffinePartial2D(
                    pts_followup, pts_baseline,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.config.ransac_reproj_threshold,
                    maxIters=self.config.ransac_max_iters,
                    confidence=self.config.ransac_confidence
                )
            elif motion_model == MotionModel.TRANSLATION:
                # For translation-only, compute median shift
                shifts = pts_baseline - pts_followup
                median_shift = np.median(shifts.reshape(-1, 2), axis=0)
                transform = np.array([
                    [1, 0, median_shift[0]],
                    [0, 1, median_shift[1]]
                ], dtype=np.float64)
                
                # Create mask based on distance from median
                distances = np.linalg.norm(shifts.reshape(-1, 2) - median_shift, axis=1)
                mask = (distances < self.config.ransac_reproj_threshold).astype(np.uint8)
            else:
                raise RegistrationError(
                    stage="transform_estimation",
                    reason=f"Unsupported motion model: {motion_model}"
                )
            
        except cv2.error as e:
            raise RegistrationError(
                stage="transform_estimation",
                reason=f"OpenCV transform estimation failed: {e}"
            )
        
        if transform is None:
            raise RegistrationError(
                stage="transform_estimation",
                reason="RANSAC failed to find valid transform",
                details="Try increasing ransac_max_iters or reducing ransac_reproj_threshold"
            )
        
        # Ensure correct shape for mask
        if mask is not None:
            mask = mask.ravel()
        
        return transform, mask
    
    def _validate_transform(self, transform: np.ndarray) -> List[str]:
        """
        Validate transform against configured limits.
        
        Returns list of warning messages for any limit violations.
        """
        warnings = []
        
        # Extract translation
        tx, ty = transform[0, 2], transform[1, 2]
        translation_magnitude = np.sqrt(tx ** 2 + ty ** 2)
        
        if translation_magnitude > self.config.max_translation_pixels:
            warnings.append(
                f"Large translation detected: {translation_magnitude:.1f} pixels "
                f"(limit: {self.config.max_translation_pixels})"
            )
        
        # Extract rotation (for non-translation models)
        if self.config.motion_model != MotionModel.TRANSLATION:
            cos_theta = transform[0, 0]
            sin_theta = transform[1, 0]
            angle_deg = np.degrees(np.arctan2(sin_theta, cos_theta))
            
            if abs(angle_deg) > self.config.max_rotation_degrees:
                warnings.append(
                    f"Large rotation detected: {angle_deg:.1f}° "
                    f"(limit: ±{self.config.max_rotation_degrees}°)"
                )
        
        # Extract scale (for affine/homography models)
        if self.config.motion_model in (MotionModel.AFFINE, MotionModel.HOMOGRAPHY):
            A = transform[:2, :2]
            U, S, Vt = np.linalg.svd(A)
            scale_x, scale_y = S[0], S[1]
            
            max_scale_deviation = max(abs(scale_x - 1), abs(scale_y - 1))
            if max_scale_deviation > self.config.max_scale_change:
                warnings.append(
                    f"Large scale change detected: ({scale_x:.3f}, {scale_y:.3f}) "
                    f"(limit: ±{self.config.max_scale_change})"
                )
        
        for warning in warnings:
            logger.warning(warning)
        
        return warnings
    
    def _compute_quality_metrics(
        self,
        pts_baseline: np.ndarray,
        pts_followup: np.ndarray,
        transform: np.ndarray,
        inlier_mask: np.ndarray
    ) -> dict:
        """Compute registration quality metrics."""
        metrics = {}
        
        # Inlier ratio
        total_matches = len(pts_baseline)
        num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else total_matches
        metrics["inlier_ratio"] = num_inliers / total_matches if total_matches > 0 else 0
        metrics["num_inliers"] = num_inliers
        metrics["num_matches"] = total_matches
        
        # Reprojection error for inliers
        if inlier_mask is not None and num_inliers > 0:
            inlier_indices = np.where(inlier_mask)[0]
            pts_src = pts_followup[inlier_indices].reshape(-1, 2)
            pts_dst = pts_baseline[inlier_indices].reshape(-1, 2)
            
            # Transform source points
            if transform.shape[0] == 3:
                # Homography
                pts_src_h = np.hstack([pts_src, np.ones((len(pts_src), 1))])
                pts_transformed = (transform @ pts_src_h.T).T
                pts_transformed = pts_transformed[:, :2] / pts_transformed[:, 2:3]
            else:
                # Affine
                pts_src_h = np.hstack([pts_src, np.ones((len(pts_src), 1))])
                pts_transformed = (transform @ pts_src_h.T).T
            
            # Compute reprojection errors
            errors = np.linalg.norm(pts_transformed - pts_dst, axis=1)
            metrics["reproj_error_mean"] = float(np.mean(errors))
            metrics["reproj_error_std"] = float(np.std(errors))
            metrics["reproj_error_max"] = float(np.max(errors))
        
        return metrics
    
    def visualize_matches(
        self,
        baseline: ImageData,
        followup: ImageData,
        feature_result: FeatureMatchResult,
        show_inliers_only: bool = True
    ) -> np.ndarray:
        """
        Create visualization of feature matches.
        
        Args:
            baseline: Baseline image
            followup: Follow-up image
            feature_result: Feature matching result
            show_inliers_only: Only show inlier matches
        
        Returns:
            Visualization image as numpy array
        """
        img_baseline = self._prepare_image(baseline)
        img_followup = self._prepare_image(followup)
        
        # Convert to color for visualization
        img_baseline = cv2.cvtColor(img_baseline, cv2.COLOR_GRAY2BGR)
        img_followup = cv2.cvtColor(img_followup, cv2.COLOR_GRAY2BGR)
        
        # Filter matches
        if show_inliers_only and feature_result.inlier_mask is not None:
            mask = feature_result.inlier_mask.ravel().tolist()
            draw_matches = [m for m, is_inlier in zip(feature_result.matches, mask) if is_inlier]
        else:
            draw_matches = feature_result.matches
        
        # Draw matches
        vis = cv2.drawMatches(
            img_baseline, feature_result.keypoints_baseline,
            img_followup, feature_result.keypoints_followup,
            draw_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return vis
