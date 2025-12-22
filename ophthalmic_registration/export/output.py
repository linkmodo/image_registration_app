"""
Output and export utilities for ophthalmic image registration.

This module provides tools for saving registered images, transforms,
metadata, and generating reports.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from ophthalmic_registration.io.dicom_writer import DicomWriter
    DICOM_WRITER_AVAILABLE = True
except ImportError:
    DICOM_WRITER_AVAILABLE = False

from ophthalmic_registration.core.image_data import (
    ImageData,
    TransformResult,
    MotionModel,
)

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class ExportManager:
    """
    Manager for exporting registration results and images.
    
    Provides comprehensive export functionality including registered
    images, transform matrices, metadata, and quality reports.
    
    Attributes:
        output_dir: Default output directory
        image_format: Default image format for exports
    
    Example:
        >>> export = ExportManager(output_dir="./results")
        >>> 
        >>> # Save registered image
        >>> export.save_image(registered, "registered_followup.png")
        >>> 
        >>> # Save transform
        >>> export.save_transform(result, "transform.json")
        >>> 
        >>> # Export complete results
        >>> export.export_registration_results(
        ...     baseline, followup, registered, result,
        ...     prefix="case_001"
        ... )
    """
    
    SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        image_format: str = ".png",
        create_dirs: bool = True
    ):
        """
        Initialize export manager.
        
        Args:
            output_dir: Default output directory
            image_format: Default image format (e.g., '.png', '.tiff')
            create_dirs: Create output directories if they don't exist
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.image_format = image_format
        self.create_dirs = create_dirs
        
        if create_dirs:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_image(
        self,
        image: ImageData,
        filename: str,
        output_dir: Optional[Path] = None,
        normalize: bool = True
    ) -> Path:
        """
        Save image to file.
        
        Args:
            image: ImageData to save
            filename: Output filename
            output_dir: Output directory (uses default if None)
            normalize: Normalize to uint8 before saving
        
        Returns:
            Path to saved file
        """
        if not CV2_AVAILABLE and not PIL_AVAILABLE:
            raise ImportError("OpenCV or Pillow required for image saving")
        
        output_dir = output_dir or self.output_dir
        output_path = output_dir / filename
        
        # Ensure directory exists
        if self.create_dirs:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare pixel data
        img = image.pixel_array
        
        if normalize and img.dtype != np.uint8:
            img = image.as_uint8()
        
        # Save using OpenCV or Pillow
        if CV2_AVAILABLE:
            # Convert RGB to BGR for OpenCV
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            
            cv2.imwrite(str(output_path), img)
        else:
            pil_image = Image.fromarray(img)
            pil_image.save(output_path)
        
        logger.info(f"Saved image to {output_path}")
        return output_path
    
    def save_dicom(
        self,
        image: ImageData,
        filename: str,
        output_dir: Optional[Path] = None,
        source_dicom: Optional[Union[str, Path]] = None,
        description: Optional[str] = None
    ) -> Path:
        """
        Save image as DICOM file.
        
        Args:
            image: ImageData to save
            filename: Output filename (should end with .dcm)
            output_dir: Output directory (uses default if None)
            source_dicom: Path to source DICOM to copy metadata from
            description: Series description to add
        
        Returns:
            Path to saved DICOM file
        
        Raises:
            ImportError: If pydicom is not available
        """
        if not DICOM_WRITER_AVAILABLE:
            raise ImportError(
                "DICOM writer not available. Ensure pydicom is installed: "
                "pip install pydicom"
            )
        
        output_dir = output_dir or self.output_dir
        output_path = output_dir / filename
        
        # Ensure directory exists
        if self.create_dirs:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use DicomWriter to save with metadata preservation
        writer = DicomWriter()
        writer.save(
            image=image,
            output_path=output_path,
            source_dicom=source_dicom,
            description=description,
            add_registration_note=True
        )
        
        logger.info(f"Saved DICOM to {output_path}")
        return output_path
    
    def save_transform(
        self,
        result: TransformResult,
        filename: str,
        output_dir: Optional[Path] = None,
        include_metrics: bool = True
    ) -> Path:
        """
        Save transform result to JSON file.
        
        Args:
            result: TransformResult to save
            filename: Output filename
            output_dir: Output directory
            include_metrics: Include quality metrics
        
        Returns:
            Path to saved file
        """
        output_dir = output_dir or self.output_dir
        output_path = output_dir / filename
        
        if self.create_dirs:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = result.to_dict()
        
        if not include_metrics:
            data.pop("quality_metrics", None)
        
        # Add export metadata
        data["export_info"] = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved transform to {output_path}")
        return output_path
    
    def load_transform(self, filepath: Union[str, Path]) -> TransformResult:
        """
        Load transform from JSON file.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            TransformResult loaded from file
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return TransformResult.from_dict(data)
    
    def save_metadata(
        self,
        image: ImageData,
        filename: str,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Save image metadata to JSON file.
        
        Args:
            image: ImageData with metadata
            filename: Output filename
            output_dir: Output directory
        
        Returns:
            Path to saved file
        """
        output_dir = output_dir or self.output_dir
        output_path = output_dir / filename
        
        if self.create_dirs:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": image.metadata.to_dict(),
            "image_info": {
                "shape": image.shape,
                "dtype": str(image.dtype),
                "is_color": image.is_color,
                "filepath": image.filepath,
                "is_preprocessed": image.is_preprocessed,
                "preprocessing_history": image.preprocessing_history,
            },
            "export_info": {
                "timestamp": datetime.now().isoformat(),
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved metadata to {output_path}")
        return output_path
    
    def export_registration_results(
        self,
        baseline: ImageData,
        followup: ImageData,
        registered: ImageData,
        result: TransformResult,
        prefix: str = "registration",
        output_dir: Optional[Path] = None,
        save_originals: bool = False,
        save_overlay: bool = True,
        save_difference: bool = True
    ) -> Dict[str, Path]:
        """
        Export complete registration results.
        
        Creates a comprehensive export including:
        - Registered image
        - Transform JSON
        - Quality report
        - Optional: original images, overlay, difference map
        
        Args:
            baseline: Baseline image
            followup: Original follow-up image
            registered: Registered follow-up image
            result: Registration result
            prefix: Filename prefix
            output_dir: Output directory
            save_originals: Also save baseline and original follow-up
            save_overlay: Create and save overlay image
            save_difference: Create and save difference map
        
        Returns:
            Dictionary mapping output type to file path
        """
        output_dir = output_dir or self.output_dir
        exports = {}
        
        # Save registered image
        exports["registered"] = self.save_image(
            registered,
            f"{prefix}_registered{self.image_format}",
            output_dir
        )
        
        # Save transform
        exports["transform"] = self.save_transform(
            result,
            f"{prefix}_transform.json",
            output_dir
        )
        
        # Save originals if requested
        if save_originals:
            exports["baseline"] = self.save_image(
                baseline,
                f"{prefix}_baseline{self.image_format}",
                output_dir
            )
            exports["followup"] = self.save_image(
                followup,
                f"{prefix}_followup{self.image_format}",
                output_dir
            )
        
        # Create and save overlay
        if save_overlay and CV2_AVAILABLE:
            overlay = self._create_overlay(baseline, registered)
            overlay_path = output_dir / f"{prefix}_overlay{self.image_format}"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            exports["overlay"] = overlay_path
        
        # Create and save difference map
        if save_difference and CV2_AVAILABLE:
            diff = self._create_difference_map(baseline, registered)
            diff_path = output_dir / f"{prefix}_difference{self.image_format}"
            cv2.imwrite(str(diff_path), diff)
            exports["difference"] = diff_path
        
        # Generate quality report
        exports["report"] = self.generate_quality_report(
            result,
            f"{prefix}_report.json",
            output_dir
        )
        
        logger.info(f"Exported {len(exports)} files to {output_dir}")
        return exports
    
    def generate_quality_report(
        self,
        result: TransformResult,
        filename: str,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Generate detailed quality report.
        
        Args:
            result: TransformResult to analyze
            filename: Output filename
            output_dir: Output directory
        
        Returns:
            Path to report file
        """
        output_dir = output_dir or self.output_dir
        output_path = output_dir / filename
        
        if self.create_dirs:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "summary": {
                "motion_model": result.motion_model.value,
                "registration_time_ms": result.registration_time_ms,
                "overall_quality": self._assess_quality(result),
            },
            "transform": {
                "translation_x": result.translation[0],
                "translation_y": result.translation[1],
                "rotation_degrees": result.rotation_degrees,
                "scale_factors": result.scale_factors,
            },
            "alignment_quality": {
                "ecc_correlation": result.ecc_correlation,
                "ecc_converged": result.ecc_converged,
            },
            "metrics": result.quality_metrics,
            "warnings": result.warnings,
            "report_info": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
            }
        }
        
        # Add feature matching info if available
        if result.feature_match_result is not None:
            report["feature_matching"] = {
                "num_keypoints_baseline": len(result.feature_match_result.keypoints_baseline),
                "num_keypoints_followup": len(result.feature_match_result.keypoints_followup),
                "num_matches": len(result.feature_match_result.matches),
                "num_inliers": result.feature_match_result.num_inliers,
                "inlier_ratio": result.feature_match_result.match_ratio,
            }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Generated quality report: {output_path}")
        return output_path
    
    def generate_text_report(
        self,
        result: TransformResult,
        filename: str,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Generate human-readable text report.
        
        Args:
            result: TransformResult to analyze
            filename: Output filename
            output_dir: Output directory
        
        Returns:
            Path to report file
        """
        output_dir = output_dir or self.output_dir
        output_path = output_dir / filename
        
        if self.create_dirs:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = [
            "=" * 60,
            "OPHTHALMIC IMAGE REGISTRATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "REGISTRATION SUMMARY",
            "-" * 40,
            f"Motion Model: {result.motion_model.value}",
            f"Processing Time: {result.registration_time_ms:.1f} ms",
            f"Overall Quality: {self._assess_quality(result).upper()}",
            "",
            "TRANSFORM PARAMETERS",
            "-" * 40,
            f"Translation: ({result.translation[0]:.2f}, {result.translation[1]:.2f}) pixels",
        ]
        
        if result.rotation_degrees is not None:
            lines.append(f"Rotation: {result.rotation_degrees:.3f}°")
        
        sx, sy = result.scale_factors
        lines.append(f"Scale Factors: ({sx:.4f}, {sy:.4f})")
        
        lines.extend([
            "",
            "ALIGNMENT QUALITY",
            "-" * 40,
        ])
        
        if result.ecc_correlation is not None:
            lines.append(f"ECC Correlation: {result.ecc_correlation:.4f}")
            lines.append(f"ECC Converged: {'Yes' if result.ecc_converged else 'No'}")
        
        if result.quality_metrics:
            lines.extend([
                "",
                "QUALITY METRICS",
                "-" * 40,
            ])
            for key, value in result.quality_metrics.items():
                if isinstance(value, float):
                    lines.append(f"{key}: {value:.4f}")
                else:
                    lines.append(f"{key}: {value}")
        
        if result.feature_match_result is not None:
            lines.extend([
                "",
                "FEATURE MATCHING",
                "-" * 40,
                f"Keypoints (Baseline): {len(result.feature_match_result.keypoints_baseline)}",
                f"Keypoints (Follow-up): {len(result.feature_match_result.keypoints_followup)}",
                f"Matches: {len(result.feature_match_result.matches)}",
                f"Inliers: {result.feature_match_result.num_inliers}",
                f"Inlier Ratio: {result.feature_match_result.match_ratio:.1%}",
            ])
        
        if result.warnings:
            lines.extend([
                "",
                "WARNINGS",
                "-" * 40,
            ])
            for warning in result.warnings:
                lines.append(f"• {warning}")
        
        lines.extend([
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ])
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        logger.info(f"Generated text report: {output_path}")
        return output_path
    
    def _assess_quality(self, result: TransformResult) -> str:
        """Assess overall registration quality."""
        issues = 0
        
        # Check ECC correlation
        if result.ecc_correlation is not None:
            if result.ecc_correlation < 0.5:
                issues += 2
            elif result.ecc_correlation < 0.7:
                issues += 1
        
        # Check convergence
        if not result.ecc_converged:
            issues += 1
        
        # Check warnings
        issues += len(result.warnings)
        
        # Check overlap
        overlap = result.quality_metrics.get("overlap_ratio", 1.0)
        if overlap < 0.5:
            issues += 2
        elif overlap < 0.7:
            issues += 1
        
        if issues == 0:
            return "excellent"
        elif issues <= 2:
            return "good"
        elif issues <= 4:
            return "acceptable"
        else:
            return "poor"
    
    def _create_overlay(
        self,
        image1: ImageData,
        image2: ImageData,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create blended overlay."""
        img1 = image1.as_uint8()
        img2 = image2.as_uint8()
        
        # Ensure same size
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
        
        # Colorize
        if img1.ndim == 2:
            img1_color = np.stack([img1, np.zeros_like(img1), img1], axis=-1)
        else:
            img1_color = img1.copy()
        
        if img2.ndim == 2:
            img2_color = np.stack([np.zeros_like(img2), img2, np.zeros_like(img2)], axis=-1)
        else:
            img2_color = img2.copy()
        
        overlay = cv2.addWeighted(img1_color, 1 - alpha, img2_color, alpha, 0)
        return overlay
    
    def _create_difference_map(
        self,
        image1: ImageData,
        image2: ImageData
    ) -> np.ndarray:
        """Create difference map."""
        img1 = image1.as_uint8()
        img2 = image2.as_uint8()
        
        # Ensure same size
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
        
        # Convert to grayscale
        if img1.ndim == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if img2.ndim == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        diff = cv2.absdiff(img1, img2)
        return diff
