"""
Unit tests for the registration pipeline.

These tests verify core registration functionality using synthetic
test images to ensure reproducible behavior.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ophthalmic_registration.core.image_data import (
    ImageData,
    ImageMetadata,
    PixelSpacing,
    TransformResult,
    RegistrationConfig,
    MotionModel,
)
from ophthalmic_registration.core.exceptions import (
    RegistrationError,
    InsufficientFeaturesError,
)


def create_test_image(size=256, seed=42):
    """Create a synthetic test image with reproducible features."""
    np.random.seed(seed)
    
    # Create base image with gradients and features
    y, x = np.ogrid[:size, :size]
    center = size // 2
    
    # Background with gradient
    img = 100 + 50 * np.exp(-((x - center)**2 + (y - center)**2) / (2 * 80**2))
    
    # Add circular features
    for i in range(5):
        cx = np.random.randint(50, size - 50)
        cy = np.random.randint(50, size - 50)
        r = np.random.randint(10, 30)
        mask = ((x - cx)**2 + (y - cy)**2) < r**2
        img[mask] = np.random.randint(150, 250)
    
    # Add noise
    img = img + np.random.normal(0, 5, (size, size))
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img


def create_transformed_image(img, tx=10, ty=5, angle=2.0):
    """Create a transformed version of an image."""
    if not CV2_AVAILABLE:
        return img
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Create rotation matrix with translation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    transformed = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return transformed


class TestImageData(unittest.TestCase):
    """Tests for ImageData class."""
    
    def test_image_data_creation(self):
        """Test basic ImageData creation."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        data = ImageData(pixel_array=img)
        
        self.assertEqual(data.shape, (100, 100))
        self.assertEqual(data.height, 100)
        self.assertEqual(data.width, 100)
        self.assertTrue(data.is_grayscale)
        self.assertFalse(data.is_color)
    
    def test_image_data_with_metadata(self):
        """Test ImageData with metadata."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        spacing = PixelSpacing(0.01, 0.01, "mm", "test")
        metadata = ImageMetadata(pixel_spacing=spacing)
        
        data = ImageData(pixel_array=img, metadata=metadata)
        
        self.assertIsNotNone(data.pixel_spacing)
        self.assertEqual(data.pixel_spacing.mean_spacing, 0.01)
    
    def test_image_data_copy(self):
        """Test ImageData copy method."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        data = ImageData(pixel_array=img)
        
        copy = data.copy()
        
        self.assertIsNot(data.pixel_array, copy.pixel_array)
        np.testing.assert_array_equal(data.pixel_array, copy.pixel_array)
    
    def test_as_uint8(self):
        """Test conversion to uint8."""
        img = np.random.rand(100, 100).astype(np.float32) * 1000
        data = ImageData(pixel_array=img)
        
        uint8_img = data.as_uint8()
        
        self.assertEqual(uint8_img.dtype, np.uint8)
        self.assertEqual(uint8_img.max(), 255)
        self.assertEqual(uint8_img.min(), 0)


class TestPixelSpacing(unittest.TestCase):
    """Tests for PixelSpacing class."""
    
    def test_pixel_spacing_creation(self):
        """Test PixelSpacing creation."""
        spacing = PixelSpacing(0.01, 0.01, "mm", "dicom")
        
        self.assertEqual(spacing.row_spacing, 0.01)
        self.assertEqual(spacing.column_spacing, 0.01)
        self.assertTrue(spacing.is_isotropic)
    
    def test_anisotropic_spacing(self):
        """Test anisotropic pixel spacing."""
        spacing = PixelSpacing(0.01, 0.02, "mm", "dicom")
        
        self.assertFalse(spacing.is_isotropic)
        self.assertEqual(spacing.mean_spacing, 0.015)
    
    def test_to_microns(self):
        """Test conversion to microns."""
        spacing = PixelSpacing(0.01, 0.01, "mm", "dicom")
        
        micron_spacing = spacing.to_microns()
        
        self.assertEqual(micron_spacing.row_spacing, 10.0)
        self.assertEqual(micron_spacing.unit, "um")


class TestTransformResult(unittest.TestCase):
    """Tests for TransformResult class."""
    
    def test_transform_result_creation(self):
        """Test TransformResult creation."""
        matrix = np.array([[1, 0, 10], [0, 1, 5]], dtype=np.float64)
        result = TransformResult(
            transform_matrix=matrix,
            motion_model=MotionModel.AFFINE
        )
        
        self.assertEqual(result.translation, (10, 5))
        self.assertEqual(result.rotation_degrees, 0.0)
    
    def test_transform_with_rotation(self):
        """Test transform with rotation."""
        angle_rad = np.radians(30)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=np.float64)
        
        result = TransformResult(
            transform_matrix=matrix,
            motion_model=MotionModel.EUCLIDEAN
        )
        
        self.assertAlmostEqual(result.rotation_degrees, 30.0, places=5)
    
    def test_identity_detection(self):
        """Test identity transform detection."""
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        result = TransformResult(
            transform_matrix=identity,
            motion_model=MotionModel.AFFINE
        )
        
        self.assertTrue(result.is_identity)
    
    def test_serialization(self):
        """Test transform serialization."""
        matrix = np.array([[1, 0, 10], [0, 1, 5]], dtype=np.float64)
        result = TransformResult(
            transform_matrix=matrix,
            motion_model=MotionModel.AFFINE,
            ecc_correlation=0.95
        )
        
        data = result.to_dict()
        loaded = TransformResult.from_dict(data)
        
        np.testing.assert_array_almost_equal(
            result.transform_matrix,
            loaded.transform_matrix
        )
        self.assertEqual(result.motion_model, loaded.motion_model)


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not available")
class TestRegistrationPipeline(unittest.TestCase):
    """Integration tests for registration pipeline."""
    
    def setUp(self):
        """Set up test images."""
        self.baseline_img = create_test_image(256, seed=42)
        self.followup_img = create_transformed_image(
            self.baseline_img, tx=15, ty=-10, angle=3.0
        )
        
        self.baseline = ImageData(pixel_array=self.baseline_img)
        self.followup = ImageData(pixel_array=self.followup_img)
    
    def test_sift_alignment(self):
        """Test SIFT coarse alignment."""
        from ophthalmic_registration.registration.sift_aligner import SiftAligner
        
        config = RegistrationConfig(motion_model=MotionModel.AFFINE)
        aligner = SiftAligner(config)
        
        result = aligner.align(self.baseline, self.followup)
        
        self.assertIsInstance(result, TransformResult)
        self.assertIsNotNone(result.feature_match_result)
        self.assertGreater(result.feature_match_result.num_inliers, 0)
    
    def test_ecc_refinement(self):
        """Test ECC fine alignment."""
        from ophthalmic_registration.registration.ecc_aligner import EccAligner
        
        config = RegistrationConfig(motion_model=MotionModel.AFFINE)
        aligner = EccAligner(config)
        
        # Start with identity
        initial = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
        result = aligner.refine(self.baseline, self.followup, initial)
        
        self.assertIsInstance(result, TransformResult)
        self.assertIsNotNone(result.ecc_correlation)
    
    def test_full_pipeline(self):
        """Test complete two-stage registration."""
        from ophthalmic_registration.registration.registration_pipeline import RegistrationPipeline
        
        config = RegistrationConfig(
            motion_model=MotionModel.AFFINE,
            use_coarse_alignment=True,
            use_fine_alignment=True
        )
        
        pipeline = RegistrationPipeline(config=config)
        result, registered = pipeline.register_and_apply(self.baseline, self.followup)
        
        self.assertIsInstance(result, TransformResult)
        self.assertIsInstance(registered, ImageData)
        self.assertGreater(result.ecc_correlation, 0.5)
        
        # Check that transform approximately recovers known transformation
        tx, ty = result.translation
        self.assertAlmostEqual(tx, -15, delta=5)  # Inverse of applied tx
        self.assertAlmostEqual(ty, 10, delta=5)   # Inverse of applied ty


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not available")
class TestPreprocessing(unittest.TestCase):
    """Tests for preprocessing pipeline."""
    
    def test_grayscale_conversion(self):
        """Test grayscale conversion."""
        from ophthalmic_registration.preprocessing.pipeline import PreprocessingPipeline, PreprocessingConfig, PreprocessingStep
        
        # Create color image
        color_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        data = ImageData(pixel_array=color_img)
        
        config = PreprocessingConfig(steps=[PreprocessingStep.GRAYSCALE])
        pipeline = PreprocessingPipeline(config)
        
        result = pipeline.process(data)
        
        self.assertEqual(result.pixel_array.ndim, 2)
        self.assertTrue(result.is_preprocessed)
    
    def test_clahe(self):
        """Test CLAHE application."""
        from ophthalmic_registration.preprocessing.pipeline import PreprocessingPipeline, PreprocessingConfig, PreprocessingStep
        
        img = create_test_image(100)
        data = ImageData(pixel_array=img)
        
        config = PreprocessingConfig(
            steps=[PreprocessingStep.CLAHE],
            clahe_clip_limit=2.0
        )
        pipeline = PreprocessingPipeline(config)
        
        result = pipeline.process(data)
        
        # CLAHE should increase contrast
        self.assertGreaterEqual(result.pixel_array.std(), img.std() * 0.8)


class TestMeasurement(unittest.TestCase):
    """Tests for spatial measurement utilities."""
    
    def test_distance_measurement(self):
        """Test distance measurement."""
        from ophthalmic_registration.measurement.spatial import SpatialCalibration
        
        spacing = PixelSpacing(0.01, 0.01, "mm", "test")
        calibration = SpatialCalibration(pixel_spacing=spacing)
        
        # 100 pixels at 0.01 mm/pixel = 1 mm
        measurement = calibration.measure_distance((0, 0), (100, 0), unit="mm")
        
        self.assertAlmostEqual(measurement.real_value, 1.0, places=5)
        self.assertEqual(measurement.unit, "mm")
    
    def test_area_measurement(self):
        """Test area measurement."""
        from ophthalmic_registration.measurement.spatial import SpatialCalibration
        
        spacing = PixelSpacing(0.01, 0.01, "mm", "test")
        calibration = SpatialCalibration(pixel_spacing=spacing)
        
        # 100x100 pixel square at 0.01 mm/pixel = 1 mm²
        points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        measurement = calibration.measure_area(points, unit="mm")
        
        self.assertAlmostEqual(measurement.real_value, 1.0, places=5)
        self.assertEqual(measurement.unit, "mm²")


if __name__ == "__main__":
    unittest.main()
