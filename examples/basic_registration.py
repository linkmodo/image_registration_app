#!/usr/bin/env python
"""
Basic Registration Example

This script demonstrates the complete workflow for registering
two ophthalmic images using the two-stage SIFT+ECC pipeline.

Usage:
    python basic_registration.py baseline.png followup.png --output ./results

Example with DICOM:
    python basic_registration.py baseline.dcm followup.dcm --output ./results
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ophthalmic_registration import (
    ImageLoader,
    PreprocessingPipeline,
    RegistrationPipeline,
    Visualizer,
    SpatialCalibration,
    ExportManager,
)
from ophthalmic_registration.core.image_data import (
    RegistrationConfig,
    MotionModel,
    PixelSpacing,
)
from ophthalmic_registration.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingStep,
)
from ophthalmic_registration.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """Main entry point for basic registration example."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Register two ophthalmic images using SIFT+ECC"
    )
    parser.add_argument(
        "baseline",
        type=str,
        help="Path to baseline (reference) image"
    )
    parser.add_argument(
        "followup",
        type=str,
        help="Path to follow-up (moving) image"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./registration_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--motion-model",
        type=str,
        choices=["translation", "euclidean", "affine", "homography"],
        default="affine",
        help="Motion model for registration"
    )
    parser.add_argument(
        "--pixel-spacing",
        type=float,
        default=None,
        help="Manual pixel spacing in mm (if not in metadata)"
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip preprocessing"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization windows"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    logger.info("=" * 60)
    logger.info("Ophthalmic Image Registration")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Load Images
    # =========================================================================
    logger.info("Step 1: Loading images...")
    
    loader = ImageLoader()
    
    # Optional pixel spacing
    pixel_spacing = None
    if args.pixel_spacing:
        pixel_spacing = PixelSpacing(
            row_spacing=args.pixel_spacing,
            column_spacing=args.pixel_spacing,
            unit="mm",
            source="manual"
        )
    
    try:
        baseline = loader.load(args.baseline, pixel_spacing=pixel_spacing)
        followup = loader.load(args.followup, pixel_spacing=pixel_spacing)
    except Exception as e:
        logger.error(f"Failed to load images: {e}")
        return 1
    
    logger.info(f"  Baseline: {baseline.shape}, dtype={baseline.dtype}")
    logger.info(f"  Follow-up: {followup.shape}, dtype={followup.dtype}")
    
    if baseline.pixel_spacing:
        logger.info(f"  Pixel spacing: {baseline.pixel_spacing.mean_spacing:.4f} mm")
    else:
        logger.warning("  No pixel spacing available - measurements will be in pixels")
    
    # =========================================================================
    # Step 2: Configure Preprocessing
    # =========================================================================
    preprocessor = None
    if not args.no_preprocess:
        logger.info("Step 2: Configuring preprocessing...")
        
        preprocess_config = PreprocessingConfig(
            steps=[
                PreprocessingStep.GRAYSCALE,
                PreprocessingStep.CLAHE,
                PreprocessingStep.GAUSSIAN_BLUR,
            ],
            clahe_clip_limit=2.0,
            clahe_tile_size=(8, 8),
            gaussian_sigma=0.5,
        )
        preprocessor = PreprocessingPipeline(preprocess_config)
        logger.info(f"  Steps: {[s.value for s in preprocess_config.steps]}")
    
    # =========================================================================
    # Step 3: Configure Registration
    # =========================================================================
    logger.info("Step 3: Configuring registration...")
    
    motion_model = MotionModel(args.motion_model)
    
    reg_config = RegistrationConfig(
        motion_model=motion_model,
        use_coarse_alignment=True,
        use_fine_alignment=True,
        sift_n_features=5000,
        match_ratio_threshold=0.75,
        ransac_reproj_threshold=5.0,
        ecc_max_iterations=1000,
        ecc_epsilon=1e-6,
        min_matches_required=10,
        validate_transform=True,
    )
    
    pipeline = RegistrationPipeline(
        config=reg_config,
        preprocessor=preprocessor
    )
    
    logger.info(f"  Motion model: {motion_model.value}")
    logger.info(f"  Coarse alignment: SIFT + RANSAC")
    logger.info(f"  Fine alignment: ECC")
    
    # =========================================================================
    # Step 4: Perform Registration
    # =========================================================================
    logger.info("Step 4: Performing registration...")
    
    try:
        result, registered = pipeline.register_and_apply(
            baseline, followup,
            preprocess=not args.no_preprocess
        )
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return 1
    
    # Log results
    logger.info("  Registration complete!")
    logger.info(f"  Time: {result.registration_time_ms:.1f} ms")
    logger.info(f"  Translation: ({result.translation[0]:.2f}, {result.translation[1]:.2f}) px")
    
    if result.rotation_degrees is not None:
        logger.info(f"  Rotation: {result.rotation_degrees:.3f}°")
    
    if result.ecc_correlation is not None:
        logger.info(f"  ECC Correlation: {result.ecc_correlation:.4f}")
        logger.info(f"  ECC Converged: {result.ecc_converged}")
    
    if result.feature_match_result:
        logger.info(f"  Feature matches: {result.feature_match_result.num_inliers} inliers")
    
    # Log warnings
    for warning in result.warnings:
        logger.warning(f"  {warning}")
    
    # =========================================================================
    # Step 5: Spatial Measurements (if calibrated)
    # =========================================================================
    if baseline.pixel_spacing:
        logger.info("Step 5: Spatial measurements...")
        
        calibration = SpatialCalibration.from_image(baseline)
        
        # Convert translation to real units
        tx_mm = calibration.pixels_to_mm(abs(result.translation[0]))
        ty_mm = calibration.pixels_to_mm(abs(result.translation[1]))
        
        logger.info(f"  Translation: ({tx_mm:.3f}, {ty_mm:.3f}) mm")
        
        # Add scale bar to registered image
        registered_with_scale = calibration.add_scale_bar(
            registered,
            length_mm=1.0,
            position="bottom_right"
        )
    else:
        registered_with_scale = registered
    
    # =========================================================================
    # Step 6: Export Results
    # =========================================================================
    logger.info("Step 6: Exporting results...")
    
    exporter = ExportManager(output_dir=output_dir)
    
    exports = exporter.export_registration_results(
        baseline=baseline,
        followup=followup,
        registered=registered_with_scale,
        result=result,
        prefix="registration",
        save_originals=True,
        save_overlay=True,
        save_difference=True
    )
    
    # Generate text report
    exporter.generate_text_report(result, "registration_report.txt", output_dir)
    
    for export_type, path in exports.items():
        logger.info(f"  Saved {export_type}: {path}")
    
    # =========================================================================
    # Step 7: Visualization (optional)
    # =========================================================================
    if args.visualize:
        logger.info("Step 7: Showing visualization...")
        
        viz = Visualizer()
        
        # Show comprehensive summary
        fig = viz.create_registration_summary(
            baseline, followup, registered, result
        )
        
        import matplotlib.pyplot as plt
        plt.show()
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Registration Complete")
    logger.info("=" * 60)
    
    quality = pipeline.get_registration_quality_summary(result)
    logger.info(f"Overall Quality: {quality['overall_quality'].upper()}")
    
    if quality['issues']:
        logger.info("Issues:")
        for issue in quality['issues']:
            logger.info(f"  - {issue}")
    
    logger.info(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
