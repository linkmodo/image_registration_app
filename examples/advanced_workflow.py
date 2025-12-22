#!/usr/bin/env python
"""
Advanced Registration Workflow Example

This script demonstrates advanced features including:
- Custom preprocessing pipelines
- Multiple motion model comparison
- Detailed quality analysis
- Longitudinal series registration
- Measurement tools

Usage:
    python advanced_workflow.py
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ophthalmic_registration import (
    ImageLoader,
    PreprocessingPipeline,
    RegistrationPipeline,
    Visualizer,
    SpatialCalibration,
    ExportManager,
    ImageData,
)
from ophthalmic_registration.core.image_data import (
    RegistrationConfig,
    MotionModel,
    PixelSpacing,
    ImageMetadata,
)
from ophthalmic_registration.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingStep,
    normalize_resolution,
)
from ophthalmic_registration.measurement.spatial import (
    Measurement,
    check_scale_compatibility,
)
from ophthalmic_registration.utils.logging_config import (
    setup_logging,
    get_logger,
    RegistrationLogger,
)

logger = get_logger(__name__)


def create_synthetic_test_images():
    """
    Create synthetic test images for demonstration.
    
    Returns:
        Tuple of (baseline, followup) ImageData objects
    """
    # Create a synthetic fundus-like image
    size = 512
    
    # Base image with circular structure (optic disc simulation)
    y, x = np.ogrid[:size, :size]
    center = size // 2
    
    # Background gradient
    background = np.zeros((size, size), dtype=np.float32)
    background = 50 + 30 * np.exp(-((x - center)**2 + (y - center)**2) / (2 * 150**2))
    
    # Optic disc
    disc_center = (center + 50, center)
    disc_radius = 40
    disc_mask = ((x - disc_center[0])**2 + (y - disc_center[1])**2) < disc_radius**2
    background[disc_mask] = 200
    
    # Add some vessel-like structures
    for i in range(5):
        angle = i * np.pi / 5
        vessel_x = center + np.cos(angle) * np.arange(200)
        vessel_y = center + np.sin(angle) * np.arange(200)
        vessel_x = vessel_x.astype(int)
        vessel_y = vessel_y.astype(int)
        valid = (vessel_x >= 0) & (vessel_x < size) & (vessel_y >= 0) & (vessel_y < size)
        background[vessel_y[valid], vessel_x[valid]] = 30
    
    # Add noise
    baseline_img = background + np.random.normal(0, 5, (size, size))
    baseline_img = np.clip(baseline_img, 0, 255).astype(np.uint8)
    
    # Create follow-up with small transformation
    # Simulate slight shift and rotation
    import cv2
    
    tx, ty = 15, -10  # Translation
    angle = 2.5  # Rotation in degrees
    
    # Create transformation matrix
    rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    rotation_matrix[0, 2] += tx
    rotation_matrix[1, 2] += ty
    
    followup_img = cv2.warpAffine(
        baseline_img, rotation_matrix, (size, size),
        borderMode=cv2.BORDER_REFLECT
    )
    
    # Add slightly different noise to follow-up
    followup_img = followup_img.astype(np.float32)
    followup_img += np.random.normal(0, 5, (size, size))
    followup_img = np.clip(followup_img, 0, 255).astype(np.uint8)
    
    # Create ImageData objects with metadata
    pixel_spacing = PixelSpacing(
        row_spacing=0.01,  # 10 microns
        column_spacing=0.01,
        unit="mm",
        source="synthetic"
    )
    
    baseline_metadata = ImageMetadata(
        pixel_spacing=pixel_spacing,
        rows=size,
        columns=size,
    )
    
    followup_metadata = ImageMetadata(
        pixel_spacing=pixel_spacing,
        rows=size,
        columns=size,
    )
    
    baseline = ImageData(
        pixel_array=baseline_img,
        metadata=baseline_metadata,
        filepath="synthetic_baseline.png"
    )
    
    followup = ImageData(
        pixel_array=followup_img,
        metadata=followup_metadata,
        filepath="synthetic_followup.png"
    )
    
    logger.info(f"Created synthetic images: {size}x{size}")
    logger.info(f"Applied transform: tx={tx}, ty={ty}, rotation={angle}°")
    
    return baseline, followup


def compare_motion_models(baseline, followup, preprocessor):
    """
    Compare registration results across different motion models.
    
    Args:
        baseline: Baseline ImageData
        followup: Follow-up ImageData
        preprocessor: Preprocessing pipeline
    
    Returns:
        Dictionary of results per motion model
    """
    logger.info("Comparing motion models...")
    
    models = [
        MotionModel.TRANSLATION,
        MotionModel.EUCLIDEAN,
        MotionModel.AFFINE,
        MotionModel.HOMOGRAPHY,
    ]
    
    results = {}
    
    for model in models:
        logger.info(f"  Testing {model.value}...")
        
        config = RegistrationConfig(
            motion_model=model,
            use_coarse_alignment=True,
            use_fine_alignment=True,
        )
        
        pipeline = RegistrationPipeline(config=config, preprocessor=preprocessor)
        
        try:
            result = pipeline.register(baseline, followup)
            results[model.value] = {
                "success": True,
                "ecc": result.ecc_correlation,
                "converged": result.ecc_converged,
                "time_ms": result.registration_time_ms,
                "translation": result.translation,
                "rotation": result.rotation_degrees,
                "result": result,
            }
            logger.info(f"    ECC: {result.ecc_correlation:.4f}, Time: {result.registration_time_ms:.1f}ms")
        except Exception as e:
            results[model.value] = {
                "success": False,
                "error": str(e),
            }
            logger.warning(f"    Failed: {e}")
    
    return results


def demonstrate_measurements(baseline, registered, result):
    """
    Demonstrate spatial measurement capabilities.
    
    Args:
        baseline: Baseline image
        registered: Registered image
        result: Registration result
    """
    logger.info("Demonstrating spatial measurements...")
    
    if not baseline.pixel_spacing:
        logger.warning("No pixel spacing - skipping measurements demo")
        return
    
    calibration = SpatialCalibration.from_image(baseline)
    
    # Measure the translation in real units
    tx, ty = result.translation
    
    distance = calibration.measure_distance(
        (0, 0),
        (abs(tx), abs(ty)),
        unit="um",
        label="Registration shift"
    )
    
    logger.info(f"  Registration shift: {distance}")
    
    # Example: measure distance between two points
    point1 = (100, 100)
    point2 = (200, 150)
    
    measurement = calibration.measure_distance(
        point1, point2,
        unit="um",
        label="Example measurement"
    )
    
    logger.info(f"  Example distance ({point1} to {point2}): {measurement}")
    
    # Example: measure area of a region
    polygon = [(100, 100), (200, 100), (200, 200), (100, 200)]
    
    area = calibration.measure_area(
        polygon,
        unit="mm",
        label="Example region"
    )
    
    logger.info(f"  Example area: {area}")
    
    # Get calibration info
    cal_info = calibration.get_calibration_info()
    logger.info(f"  Calibration: {cal_info['mean_spacing']:.4f} mm/pixel")


def main():
    """Main entry point for advanced workflow example."""
    
    # Setup logging
    setup_logging(level="INFO")
    
    logger.info("=" * 60)
    logger.info("Advanced Registration Workflow Demo")
    logger.info("=" * 60)
    
    # Create session logger
    session_logger = RegistrationLogger("demo_session")
    session_logger.start_registration()
    
    # =========================================================================
    # Create Test Images
    # =========================================================================
    logger.info("\n1. Creating synthetic test images...")
    
    try:
        baseline, followup = create_synthetic_test_images()
        session_logger.log_image_loaded("baseline", baseline.shape, baseline.pixel_spacing is not None)
        session_logger.log_image_loaded("followup", followup.shape, followup.pixel_spacing is not None)
    except ImportError:
        logger.error("OpenCV required for this demo. Install with: pip install opencv-python")
        return 1
    
    # Check scale compatibility
    compatible, details = check_scale_compatibility(baseline, followup)
    logger.info(f"  Scale compatibility: {details['message']}")
    
    # =========================================================================
    # Configure Preprocessing
    # =========================================================================
    logger.info("\n2. Configuring preprocessing pipeline...")
    
    preprocess_config = PreprocessingConfig(
        steps=[
            PreprocessingStep.GRAYSCALE,
            PreprocessingStep.NORMALIZE_INTENSITY,
            PreprocessingStep.CLAHE,
            PreprocessingStep.GAUSSIAN_BLUR,
        ],
        clahe_clip_limit=2.5,
        clahe_tile_size=(8, 8),
        gaussian_sigma=0.8,
    )
    
    preprocessor = PreprocessingPipeline(preprocess_config)
    logger.info(f"  Steps: {[s.value for s in preprocess_config.steps]}")
    
    # Apply preprocessing to see effect
    baseline_proc = preprocessor.process(baseline)
    session_logger.log_preprocessing(baseline_proc.preprocessing_history)
    
    # =========================================================================
    # Compare Motion Models
    # =========================================================================
    logger.info("\n3. Comparing motion models...")
    
    model_results = compare_motion_models(baseline, followup, preprocessor)
    
    # Find best model
    best_model = None
    best_ecc = -1
    
    for model_name, res in model_results.items():
        if res["success"] and res["ecc"] and res["ecc"] > best_ecc:
            best_ecc = res["ecc"]
            best_model = model_name
    
    logger.info(f"\n  Best model: {best_model} (ECC: {best_ecc:.4f})")
    
    # =========================================================================
    # Perform Final Registration
    # =========================================================================
    logger.info("\n4. Performing final registration with best model...")
    
    final_config = RegistrationConfig(
        motion_model=MotionModel(best_model),
        use_coarse_alignment=True,
        use_fine_alignment=True,
        sift_n_features=5000,
        ecc_max_iterations=1000,
    )
    
    final_pipeline = RegistrationPipeline(
        config=final_config,
        preprocessor=preprocessor
    )
    
    result, registered = final_pipeline.register_and_apply(baseline, followup)
    
    session_logger.log_coarse_alignment(
        num_keypoints=len(result.feature_match_result.keypoints_baseline) if result.feature_match_result else 0,
        num_matches=len(result.feature_match_result.matches) if result.feature_match_result else 0,
        num_inliers=result.feature_match_result.num_inliers if result.feature_match_result else 0,
    )
    session_logger.log_fine_alignment(result.ecc_correlation or 0, result.ecc_converged)
    
    logger.info(f"  Final ECC: {result.ecc_correlation:.4f}")
    logger.info(f"  Translation: ({result.translation[0]:.2f}, {result.translation[1]:.2f}) px")
    logger.info(f"  Rotation: {result.rotation_degrees:.3f}°")
    
    # =========================================================================
    # Quality Analysis
    # =========================================================================
    logger.info("\n5. Analyzing registration quality...")
    
    quality_summary = final_pipeline.get_registration_quality_summary(result)
    
    logger.info(f"  Overall quality: {quality_summary['overall_quality']}")
    logger.info(f"  ECC quality: {quality_summary.get('ecc_quality', 'N/A')}")
    
    if quality_summary['issues']:
        logger.info("  Issues:")
        for issue in quality_summary['issues']:
            logger.info(f"    - {issue}")
    
    if quality_summary['recommendations']:
        logger.info("  Recommendations:")
        for rec in quality_summary['recommendations']:
            logger.info(f"    - {rec}")
    
    # =========================================================================
    # Spatial Measurements
    # =========================================================================
    logger.info("\n6. Spatial measurements...")
    
    demonstrate_measurements(baseline, registered, result)
    
    # =========================================================================
    # Export Results
    # =========================================================================
    logger.info("\n7. Exporting results...")
    
    output_dir = Path("./advanced_demo_results")
    output_dir.mkdir(exist_ok=True)
    
    exporter = ExportManager(output_dir=output_dir)
    
    # Export with all options
    exports = exporter.export_registration_results(
        baseline=baseline,
        followup=followup,
        registered=registered,
        result=result,
        prefix="demo",
        save_originals=True,
        save_overlay=True,
        save_difference=True,
    )
    
    # Generate reports
    exporter.generate_text_report(result, "demo_report.txt", output_dir)
    
    logger.info(f"  Exported {len(exports)} files to {output_dir}")
    
    # =========================================================================
    # Visualization
    # =========================================================================
    logger.info("\n8. Creating visualizations...")
    
    viz = Visualizer()
    
    # Create summary figure (saved to file)
    fig = viz.create_registration_summary(baseline, followup, registered, result)
    
    import matplotlib.pyplot as plt
    fig.savefig(output_dir / "demo_summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"  Saved summary figure")
    
    # =========================================================================
    # Session Complete
    # =========================================================================
    session_logger.end_registration(success=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("Advanced Workflow Demo Complete")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir.absolute()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
