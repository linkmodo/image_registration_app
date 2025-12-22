"""
SimpleITK/Elastix-based registration for medical images.

This module provides registration using SimpleITK with optional Elastix
support. Elastix is the gold standard for longitudinal and cross-modality
registration in medical imaging.

SimpleITK is free and open source (Apache 2.0 license).
"""

import logging
import time
from typing import Optional, Tuple, Dict, Any
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


class ElastixTransformType(Enum):
    """Available Elastix transform types."""
    TRANSLATION = "TranslationTransform"
    RIGID = "EulerTransform"
    SIMILARITY = "SimilarityTransform"
    AFFINE = "AffineTransform"
    BSPLINE = "BSplineTransform"


class ElastixMetric(Enum):
    """Available similarity metrics."""
    MEAN_SQUARES = "AdvancedMeanSquares"
    NORMALIZED_CORRELATION = "AdvancedNormalizedCorrelation"
    MUTUAL_INFORMATION = "AdvancedMattesMutualInformation"


class ElastixAligner:
    """
    SimpleITK/Elastix-based image registration.
    
    Provides robust medical image registration using SimpleITK's
    registration framework with optional Elastix backend.
    
    Features:
    - Multiple transform types (rigid, affine, B-spline)
    - Multiple similarity metrics (MI, NCC, MSE)
    - Multi-resolution pyramid for robustness
    - Suitable for longitudinal and cross-modality registration
    
    Example:
        >>> aligner = ElastixAligner(transform_type=ElastixTransformType.AFFINE)
        >>> result = aligner.align(baseline_image, followup_image)
    """
    
    def __init__(
        self,
        config: Optional[RegistrationConfig] = None,
        transform_type: ElastixTransformType = ElastixTransformType.AFFINE,
        metric: ElastixMetric = ElastixMetric.MUTUAL_INFORMATION,
        use_elastix: bool = True,
        num_iterations: int = 500,
        num_resolutions: int = 4,
        learning_rate: float = 1.0,
        sampling_percentage: float = 0.1
    ):
        """
        Initialize the Elastix aligner.
        
        Args:
            config: Registration configuration
            transform_type: Type of geometric transform
            metric: Similarity metric to optimize
            use_elastix: Use Elastix backend if available (vs SimpleITK default)
            num_iterations: Maximum iterations per resolution level
            num_resolutions: Number of multi-resolution levels
            learning_rate: Optimizer learning rate
            sampling_percentage: Fraction of pixels to sample for metric
        
        Raises:
            ImportError: If SimpleITK is not available
        """
        if not SITK_AVAILABLE:
            raise ImportError(
                "SimpleITK is required for Elastix registration. "
                "Install with: pip install SimpleITK"
            )
        
        self.config = config or RegistrationConfig()
        self.transform_type = transform_type
        self.metric = metric
        self.use_elastix = use_elastix
        self.num_iterations = num_iterations
        self.num_resolutions = num_resolutions
        self.learning_rate = learning_rate
        self.sampling_percentage = sampling_percentage
        
        # Check if Elastix is available
        self._elastix_available = self._check_elastix()
        if use_elastix and not self._elastix_available:
            logger.warning(
                "Elastix not available, falling back to SimpleITK registration"
            )
    
    def _check_elastix(self) -> bool:
        """Check if Elastix is available in SimpleITK."""
        try:
            # Try to create an Elastix image filter
            elastix = sitk.ElastixImageFilter()
            return True
        except AttributeError:
            return False
        except Exception:
            return False
    
    def align(
        self,
        baseline: ImageData,
        followup: ImageData,
        initial_transform: Optional[np.ndarray] = None
    ) -> TransformResult:
        """
        Perform Elastix/SimpleITK registration.
        
        Args:
            baseline: Reference/baseline image (fixed)
            followup: Moving/follow-up image to align
            initial_transform: Optional initial transform estimate
        
        Returns:
            TransformResult with computed transformation
        
        Raises:
            RegistrationError: If registration fails
        """
        start_time = time.time()
        warnings = []
        
        # Convert to SimpleITK images
        fixed_image = self._to_sitk_image(baseline)
        moving_image = self._to_sitk_image(followup)
        
        # Ensure same size
        if fixed_image.GetSize() != moving_image.GetSize():
            logger.warning(
                f"Image sizes differ: fixed={fixed_image.GetSize()}, "
                f"moving={moving_image.GetSize()}. Resampling moving image."
            )
            moving_image = self._resample_to_reference(moving_image, fixed_image)
            warnings.append("Moving image was resampled to match fixed image size")
        
        try:
            if self.use_elastix and self._elastix_available:
                result_image, transform_params = self._run_elastix(
                    fixed_image, moving_image, initial_transform
                )
            else:
                result_image, transform_params = self._run_sitk_registration(
                    fixed_image, moving_image, initial_transform
                )
        except Exception as e:
            raise RegistrationError(
                stage="elastix_registration",
                reason=f"Registration failed: {str(e)}"
            )
        
        # Convert SimpleITK transform to OpenCV matrix
        transform_matrix = self._sitk_to_cv_transform(transform_params, baseline)
        
        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(
            fixed_image, moving_image, result_image
        )
        quality_metrics["method"] = "elastix" if self._elastix_available else "sitk"
        quality_metrics["transform_type"] = self.transform_type.value
        quality_metrics["metric"] = self.metric.value
        
        elapsed_time = (time.time() - start_time) * 1000
        
        return TransformResult(
            transform_matrix=transform_matrix,
            motion_model=self._get_motion_model(),
            coarse_transform=transform_matrix.copy(),
            registration_time_ms=elapsed_time,
            quality_metrics=quality_metrics,
            warnings=warnings
        )
    
    def _to_sitk_image(self, image_data: ImageData) -> sitk.Image:
        """Convert ImageData to SimpleITK image."""
        img = image_data.pixel_array
        
        # Convert to grayscale if needed
        if img.ndim == 3:
            if CV2_AVAILABLE:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img = np.mean(img, axis=2).astype(img.dtype)
        
        # Ensure float32 for registration
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        
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
    
    def _resample_to_reference(
        self,
        moving: sitk.Image,
        reference: sitk.Image
    ) -> sitk.Image:
        """Resample moving image to reference space."""
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(sitk.Transform())
        return resampler.Execute(moving)
    
    def _run_elastix(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        initial_transform: Optional[np.ndarray]
    ) -> Tuple[sitk.Image, Dict[str, Any]]:
        """Run Elastix registration."""
        elastix = sitk.ElastixImageFilter()
        elastix.SetFixedImage(fixed)
        elastix.SetMovingImage(moving)
        
        # Create parameter map
        param_map = self._create_elastix_parameter_map()
        elastix.SetParameterMap(param_map)
        
        # Set initial transform if provided
        if initial_transform is not None:
            init_transform = self._cv_to_sitk_transform(initial_transform, fixed)
            elastix.SetInitialTransform(init_transform)
        
        # Run registration
        elastix.Execute()
        
        # Get results
        result_image = elastix.GetResultImage()
        transform_params = elastix.GetTransformParameterMap()[0]
        
        return result_image, transform_params
    
    def _run_sitk_registration(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        initial_transform: Optional[np.ndarray]
    ) -> Tuple[sitk.Image, sitk.Transform]:
        """Run SimpleITK registration (fallback when Elastix unavailable)."""
        # Initialize registration method
        registration = sitk.ImageRegistrationMethod()
        
        # Set metric
        if self.metric == ElastixMetric.MUTUAL_INFORMATION:
            registration.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=50
            )
        elif self.metric == ElastixMetric.NORMALIZED_CORRELATION:
            registration.SetMetricAsCorrelation()
        else:
            registration.SetMetricAsMeanSquares()
        
        # Set sampling
        registration.SetMetricSamplingStrategy(
            registration.RANDOM
        )
        registration.SetMetricSamplingPercentage(self.sampling_percentage)
        
        # Set interpolator
        registration.SetInterpolator(sitk.sitkLinear)
        
        # Set optimizer
        registration.SetOptimizerAsGradientDescent(
            learningRate=self.learning_rate,
            numberOfIterations=self.num_iterations,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        
        # Set multi-resolution
        shrink_factors = [2 ** i for i in range(self.num_resolutions - 1, -1, -1)]
        smoothing_sigmas = [float(f) for f in shrink_factors]
        registration.SetShrinkFactorsPerLevel(shrink_factors)
        registration.SetSmoothingSigmasPerLevel(smoothing_sigmas)
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        # Set initial transform
        if initial_transform is not None:
            init_sitk = self._cv_to_sitk_transform(initial_transform, fixed)
            registration.SetInitialTransform(init_sitk, inPlace=False)
        else:
            init_transform = self._create_initial_transform(fixed)
            registration.SetInitialTransform(init_transform, inPlace=False)
        
        # Run registration
        final_transform = registration.Execute(fixed, moving)
        
        # Apply transform to get result image
        result_image = sitk.Resample(
            moving, fixed, final_transform,
            sitk.sitkLinear, 0.0, moving.GetPixelID()
        )
        
        return result_image, final_transform
    
    def _create_initial_transform(self, fixed: sitk.Image) -> sitk.Transform:
        """Create initial transform based on transform type."""
        if self.transform_type == ElastixTransformType.TRANSLATION:
            return sitk.TranslationTransform(2)
        elif self.transform_type == ElastixTransformType.RIGID:
            transform = sitk.Euler2DTransform()
            transform.SetCenter(self._get_image_center(fixed))
            return transform
        elif self.transform_type == ElastixTransformType.SIMILARITY:
            transform = sitk.Similarity2DTransform()
            transform.SetCenter(self._get_image_center(fixed))
            return transform
        elif self.transform_type == ElastixTransformType.AFFINE:
            transform = sitk.AffineTransform(2)
            transform.SetCenter(self._get_image_center(fixed))
            return transform
        else:
            # B-spline - use affine as initial
            transform = sitk.AffineTransform(2)
            transform.SetCenter(self._get_image_center(fixed))
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
    
    def _create_elastix_parameter_map(self):
        """Create Elastix parameter map."""
        param_map = sitk.GetDefaultParameterMap(self.transform_type.value)
        
        # Set metric
        param_map["Metric"] = [self.metric.value]
        
        # Set optimizer parameters
        param_map["MaximumNumberOfIterations"] = [str(self.num_iterations)]
        param_map["NumberOfResolutions"] = [str(self.num_resolutions)]
        
        # Set sampling
        param_map["NumberOfSpatialSamples"] = ["4096"]
        param_map["NewSamplesEveryIteration"] = ["true"]
        param_map["ImageSampler"] = ["RandomCoordinate"]
        
        # Set interpolator
        param_map["Interpolator"] = ["LinearInterpolator"]
        param_map["ResampleInterpolator"] = ["FinalLinearInterpolator"]
        
        # Set output
        param_map["WriteResultImage"] = ["false"]
        
        return param_map
    
    def _sitk_to_cv_transform(
        self,
        transform_params: Any,
        reference: ImageData
    ) -> np.ndarray:
        """Convert SimpleITK transform to OpenCV matrix."""
        h, w = reference.height, reference.width
        
        if isinstance(transform_params, dict):
            # Elastix parameter map
            params = [float(p) for p in transform_params.get("TransformParameters", ["0"] * 6)]
            transform_type = transform_params.get("Transform", ["AffineTransform"])[0]
            
            if "Translation" in transform_type:
                # Translation only
                tx, ty = params[0], params[1]
                return np.array([
                    [1, 0, tx],
                    [0, 1, ty]
                ], dtype=np.float64)
            elif "Euler" in transform_type or "Rigid" in transform_type:
                # Rigid: angle, tx, ty
                angle, tx, ty = params[0], params[1], params[2]
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                return np.array([
                    [cos_a, -sin_a, tx],
                    [sin_a, cos_a, ty]
                ], dtype=np.float64)
            elif "Similarity" in transform_type:
                # Similarity: scale, angle, tx, ty
                scale, angle, tx, ty = params[0], params[1], params[2], params[3]
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                return np.array([
                    [scale * cos_a, -scale * sin_a, tx],
                    [scale * sin_a, scale * cos_a, ty]
                ], dtype=np.float64)
            else:
                # Affine: a11, a12, a21, a22, tx, ty
                if len(params) >= 6:
                    return np.array([
                        [params[0], params[1], params[4]],
                        [params[2], params[3], params[5]]
                    ], dtype=np.float64)
        
        elif isinstance(transform_params, sitk.Transform):
            # Handle CompositeTransform by getting the last transform
            if transform_params.GetName() == "CompositeTransform":
                # Get the number of transforms in the composite
                n_transforms = transform_params.GetNumberOfTransforms()
                if n_transforms > 0:
                    # Get the last (most refined) transform
                    transform_params = transform_params.GetNthTransform(n_transforms - 1)
            
            # SimpleITK transform object
            try:
                params = transform_params.GetParameters()
                dim = transform_params.GetDimension() if hasattr(transform_params, 'GetDimension') else 2
                
                if dim == 2:
                    transform_name = transform_params.GetName() if hasattr(transform_params, 'GetName') else ""
                    
                    if "Translation" in transform_name or len(params) == 2:
                        return np.array([
                            [1, 0, params[0]],
                            [0, 1, params[1]]
                        ], dtype=np.float64)
                        
                    elif len(params) == 3:
                        # Euler2D: angle, tx, ty
                        angle, tx, ty = params
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        return np.array([
                            [cos_a, -sin_a, tx],
                            [sin_a, cos_a, ty]
                        ], dtype=np.float64)
                        
                    elif len(params) == 4:
                        # Similarity2D: scale, angle, tx, ty
                        scale, angle, tx, ty = params
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        return np.array([
                            [scale * cos_a, -scale * sin_a, tx],
                            [scale * sin_a, scale * cos_a, ty]
                        ], dtype=np.float64)
                        
                    elif len(params) >= 6:
                        # Affine - try to get matrix and translation
                        try:
                            matrix = np.array(transform_params.GetMatrix()).reshape(2, 2)
                            translation = np.array(transform_params.GetTranslation())
                            return np.hstack([matrix, translation.reshape(2, 1)])
                        except:
                            # Fallback: extract from parameters directly
                            # Affine params: a11, a12, a21, a22, tx, ty
                            return np.array([
                                [params[0], params[1], params[4]],
                                [params[2], params[3], params[5]]
                            ], dtype=np.float64)
            except Exception as e:
                logger.warning(f"Failed to extract transform parameters: {e}")
        
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
            # Affine transform
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
    
    def _get_motion_model(self) -> MotionModel:
        """Get corresponding MotionModel for transform type."""
        mapping = {
            ElastixTransformType.TRANSLATION: MotionModel.TRANSLATION,
            ElastixTransformType.RIGID: MotionModel.EUCLIDEAN,
            ElastixTransformType.SIMILARITY: MotionModel.EUCLIDEAN,
            ElastixTransformType.AFFINE: MotionModel.AFFINE,
            ElastixTransformType.BSPLINE: MotionModel.AFFINE,
        }
        return mapping.get(self.transform_type, MotionModel.AFFINE)
    
    def _compute_quality_metrics(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        result: sitk.Image
    ) -> Dict[str, Any]:
        """Compute registration quality metrics."""
        metrics = {}
        
        try:
            # Convert to arrays for comparison
            fixed_arr = sitk.GetArrayFromImage(fixed).astype(np.float64)
            result_arr = sitk.GetArrayFromImage(result).astype(np.float64)
            
            # Normalize
            fixed_arr = (fixed_arr - fixed_arr.mean()) / (fixed_arr.std() + 1e-8)
            result_arr = (result_arr - result_arr.mean()) / (result_arr.std() + 1e-8)
            
            # Normalized Cross Correlation
            ncc = np.mean(fixed_arr * result_arr)
            metrics["ncc"] = float(ncc)
            
            # Mean Squared Error
            mse = np.mean((fixed_arr - result_arr) ** 2)
            metrics["mse"] = float(mse)
            
            # Structural Similarity (simplified)
            c1, c2 = 0.01 ** 2, 0.03 ** 2
            mu_f, mu_r = fixed_arr.mean(), result_arr.mean()
            sigma_f, sigma_r = fixed_arr.std(), result_arr.std()
            sigma_fr = np.mean((fixed_arr - mu_f) * (result_arr - mu_r))
            
            ssim = ((2 * mu_f * mu_r + c1) * (2 * sigma_fr + c2)) / \
                   ((mu_f ** 2 + mu_r ** 2 + c1) * (sigma_f ** 2 + sigma_r ** 2 + c2))
            metrics["ssim"] = float(ssim)
            
        except Exception as e:
            logger.warning(f"Failed to compute quality metrics: {e}")
        
        return metrics


def is_simpleitk_available() -> bool:
    """Check if SimpleITK is available."""
    return SITK_AVAILABLE


def is_elastix_available() -> bool:
    """Check if Elastix is available in SimpleITK."""
    if not SITK_AVAILABLE:
        return False
    try:
        elastix = sitk.ElastixImageFilter()
        return True
    except:
        return False
