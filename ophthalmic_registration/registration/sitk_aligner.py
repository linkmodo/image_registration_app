"""
SimpleITK-based registration for medical images.

This module provides registration using SimpleITK's built-in registration
framework without requiring Elastix. Uses gradient descent optimization
with mutual information or mean squares metrics.

SimpleITK is free and open source (Apache 2.0 license).
"""

import logging
import time
from typing import Optional, Tuple
from enum import Enum
import numpy as np

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

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
from ophthalmic_registration.core.exceptions import RegistrationError

logger = logging.getLogger(__name__)


def is_simpleitk_available() -> bool:
    """Check if SimpleITK is available."""
    return SITK_AVAILABLE


class SitkTransformType(Enum):
    """Available SimpleITK transform types."""
    TRANSLATION = "translation"
    RIGID = "rigid"
    SIMILARITY = "similarity"
    AFFINE = "affine"


class SitkMetric(Enum):
    """Available similarity metrics."""
    MEAN_SQUARES = "mean_squares"
    CORRELATION = "correlation"
    MUTUAL_INFORMATION = "mutual_information"


class SimpleITKAligner:
    """
    SimpleITK-based image registration.
    
    Provides robust medical image registration using SimpleITK's
    registration framework with gradient descent optimization.
    
    Features:
    - Multiple transform types (translation, rigid, similarity, affine)
    - Multiple similarity metrics (mean squares, correlation, MI)
    - Multi-resolution pyramid for robustness
    - Automatic initialization
    
    Example:
        >>> aligner = SimpleITKAligner(transform_type=SitkTransformType.AFFINE)
        >>> result = aligner.align(baseline, followup)
    """
    
    def __init__(
        self,
        config: Optional[RegistrationConfig] = None,
        transform_type: SitkTransformType = SitkTransformType.AFFINE,
        metric: SitkMetric = SitkMetric.MUTUAL_INFORMATION,
        num_iterations: int = 200,
        num_resolutions: int = 3,
        learning_rate: float = 1.0,
        min_step: float = 0.001,
        relaxation_factor: float = 0.5
    ):
        """
        Initialize SimpleITK aligner.
        
        Args:
            config: Registration configuration
            transform_type: Type of transform to estimate
            metric: Similarity metric to use
            num_iterations: Maximum iterations per resolution level
            num_resolutions: Number of multi-resolution levels
            learning_rate: Initial step size for optimizer
            min_step: Minimum step size (convergence criterion)
            relaxation_factor: Step size reduction factor
        """
        if not SITK_AVAILABLE:
            raise ImportError(
                "SimpleITK is required for SimpleITKAligner. "
                "Install with: pip install SimpleITK"
            )
        
        self.config = config or RegistrationConfig()
        self.transform_type = transform_type
        self.metric = metric
        self.num_iterations = num_iterations
        self.num_resolutions = num_resolutions
        self.learning_rate = learning_rate
        self.min_step = min_step
        self.relaxation_factor = relaxation_factor
        
        # Registration state
        self._iteration_count = 0
        self._metric_values = []
    
    def align(
        self,
        baseline: ImageData,
        followup: ImageData,
        initial_transform: Optional[np.ndarray] = None
    ) -> TransformResult:
        """
        Align follow-up image to baseline using SimpleITK registration.
        
        Args:
            baseline: Reference/fixed image
            followup: Moving image to align
            initial_transform: Optional initial transform matrix
        
        Returns:
            TransformResult with transform matrix and metrics
        """
        start_time = time.time()
        warnings = []
        
        # Convert to SimpleITK images
        fixed_image = self._to_sitk_image(baseline)
        moving_image = self._to_sitk_image(followup)
        
        # Set up registration
        registration = sitk.ImageRegistrationMethod()
        
        # Set metric
        if self.metric == SitkMetric.MEAN_SQUARES:
            registration.SetMetricAsMeanSquares()
        elif self.metric == SitkMetric.CORRELATION:
            registration.SetMetricAsCorrelation()
        else:  # MUTUAL_INFORMATION
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        
        # Set optimizer - Regular Step Gradient Descent
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=self.learning_rate,
            minStep=self.min_step,
            numberOfIterations=self.num_iterations,
            relaxationFactor=self.relaxation_factor
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        
        # Set multi-resolution
        shrink_factors = [2 ** i for i in range(self.num_resolutions - 1, -1, -1)]
        smoothing_sigmas = [float(s) for s in shrink_factors]
        registration.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
        registration.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
        
        # Set interpolator
        registration.SetInterpolator(sitk.sitkLinear)
        
        # Initialize transform
        initial_sitk_transform = self._create_initial_transform(
            fixed_image, moving_image, initial_transform
        )
        registration.SetInitialTransform(initial_sitk_transform, inPlace=False)
        
        # Add iteration callback for monitoring
        self._iteration_count = 0
        self._metric_values = []
        registration.AddCommand(
            sitk.sitkIterationEvent,
            lambda: self._iteration_callback(registration)
        )
        
        try:
            # Execute registration
            final_transform = registration.Execute(fixed_image, moving_image)
            
            # Get final metric value
            final_metric = registration.GetMetricValue()
            
            logger.info(
                f"SimpleITK registration complete: "
                f"{self._iteration_count} iterations, "
                f"final metric={final_metric:.6f}"
            )
            
        except Exception as e:
            logger.error(f"SimpleITK registration failed: {e}")
            raise RegistrationError(
                stage="sitk_registration",
                reason=str(e)
            )
        
        # Convert to OpenCV transform matrix
        transform_matrix = self._sitk_to_cv_transform(final_transform, baseline)
        
        elapsed_time = (time.time() - start_time) * 1000
        
        # Build result
        result = TransformResult(
            transform_matrix=transform_matrix,
            motion_model=self._get_motion_model(),
            coarse_transform=transform_matrix.copy(),
            registration_time_ms=elapsed_time,
            quality_metrics={
                "sitk_metric": abs(final_metric),
                "iterations": self._iteration_count,
                "method": "simpleitk",
                "transform_type": self.transform_type.value,
            },
            warnings=warnings
        )
        
        return result
    
    def _iteration_callback(self, registration) -> None:
        """Callback for each iteration."""
        self._iteration_count += 1
        self._metric_values.append(registration.GetMetricValue())
    
    def _to_sitk_image(self, image_data: ImageData) -> sitk.Image:
        """Convert ImageData to SimpleITK image."""
        img = image_data.as_float32()
        
        # Convert to grayscale if needed
        if img.ndim == 3:
            if CV2_AVAILABLE:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img = np.mean(img, axis=2)
        
        # Normalize to 0-1
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        # Create SimpleITK image
        sitk_image = sitk.GetImageFromArray(img)
        
        # Set spacing if available
        if image_data.pixel_spacing:
            ps = image_data.pixel_spacing
            if hasattr(ps, 'row_spacing'):
                sitk_image.SetSpacing([ps.column_spacing, ps.row_spacing])
            elif isinstance(ps, dict):
                sitk_image.SetSpacing([
                    ps.get('column_spacing', 1.0),
                    ps.get('row_spacing', 1.0)
                ])
        
        return sitk_image
    
    def _create_initial_transform(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        cv_transform: Optional[np.ndarray] = None
    ) -> sitk.Transform:
        """Create initial transform for registration."""
        if cv_transform is not None:
            return self._cv_to_sitk_transform(cv_transform, fixed)
        
        # Use centered transform initializer
        if self.transform_type == SitkTransformType.TRANSLATION:
            transform = sitk.TranslationTransform(2)
        elif self.transform_type == SitkTransformType.RIGID:
            transform = sitk.Euler2DTransform()
        elif self.transform_type == SitkTransformType.SIMILARITY:
            transform = sitk.Similarity2DTransform()
        else:  # AFFINE
            transform = sitk.AffineTransform(2)
        
        # Center the transform
        transform = sitk.CenteredTransformInitializer(
            fixed, moving, transform,
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        
        return transform
    
    def _sitk_to_cv_transform(
        self,
        sitk_transform: sitk.Transform,
        reference: ImageData
    ) -> np.ndarray:
        """Convert SimpleITK transform to OpenCV matrix."""
        # Handle composite transforms
        if sitk_transform.GetName() == "CompositeTransform":
            n_transforms = sitk_transform.GetNumberOfTransforms()
            if n_transforms > 0:
                sitk_transform = sitk_transform.GetNthTransform(n_transforms - 1)
        
        try:
            params = sitk_transform.GetParameters()
            transform_name = sitk_transform.GetName()
            
            if "Translation" in transform_name:
                # Translation only
                return np.array([
                    [1, 0, params[0]],
                    [0, 1, params[1]]
                ], dtype=np.float64)
                
            elif "Euler2D" in transform_name:
                # Rigid: angle, tx, ty
                angle = params[0]
                tx, ty = params[1], params[2]
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                return np.array([
                    [cos_a, -sin_a, tx],
                    [sin_a, cos_a, ty]
                ], dtype=np.float64)
                
            elif "Similarity2D" in transform_name:
                # Similarity: scale, angle, tx, ty
                scale = params[0]
                angle = params[1]
                tx, ty = params[2], params[3]
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                return np.array([
                    [scale * cos_a, -scale * sin_a, tx],
                    [scale * sin_a, scale * cos_a, ty]
                ], dtype=np.float64)
                
            elif "Affine" in transform_name:
                # Affine: matrix elements + translation
                try:
                    matrix = np.array(sitk_transform.GetMatrix()).reshape(2, 2)
                    translation = np.array(sitk_transform.GetTranslation())
                    return np.hstack([matrix, translation.reshape(2, 1)])
                except:
                    # Fallback: params are a11, a12, a21, a22, tx, ty
                    if len(params) >= 6:
                        return np.array([
                            [params[0], params[1], params[4]],
                            [params[2], params[3], params[5]]
                        ], dtype=np.float64)
                        
        except Exception as e:
            logger.warning(f"Failed to convert transform: {e}")
        
        # Default: identity
        logger.warning("Could not convert transform, returning identity")
        return np.array([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float64)
    
    def _cv_to_sitk_transform(
        self,
        cv_transform: np.ndarray,
        reference: sitk.Image
    ) -> sitk.Transform:
        """Convert OpenCV matrix to SimpleITK transform."""
        center = self._get_image_center(reference)
        
        if cv_transform.shape[0] == 2:
            transform = sitk.AffineTransform(2)
            transform.SetMatrix([
                cv_transform[0, 0], cv_transform[0, 1],
                cv_transform[1, 0], cv_transform[1, 1]
            ])
            transform.SetTranslation([cv_transform[0, 2], cv_transform[1, 2]])
            transform.SetCenter(center)
            return transform
        else:
            # Homography - approximate with affine
            transform = sitk.AffineTransform(2)
            transform.SetMatrix([
                cv_transform[0, 0], cv_transform[0, 1],
                cv_transform[1, 0], cv_transform[1, 1]
            ])
            transform.SetTranslation([cv_transform[0, 2], cv_transform[1, 2]])
            transform.SetCenter(center)
            return transform
    
    def _get_image_center(self, image: sitk.Image) -> Tuple[float, float]:
        """Get physical center of image."""
        size = image.GetSize()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        center = [
            origin[i] + spacing[i] * size[i] / 2.0
            for i in range(2)
        ]
        return tuple(center)
    
    def _get_motion_model(self) -> MotionModel:
        """Get corresponding MotionModel for transform type."""
        mapping = {
            SitkTransformType.TRANSLATION: MotionModel.TRANSLATION,
            SitkTransformType.RIGID: MotionModel.EUCLIDEAN,
            SitkTransformType.SIMILARITY: MotionModel.EUCLIDEAN,
            SitkTransformType.AFFINE: MotionModel.AFFINE,
        }
        return mapping.get(self.transform_type, MotionModel.AFFINE)
