"""
Unified image I/O interface for ophthalmic images.

This module provides a single entry point for loading images from
various formats (DICOM and standard formats), automatically detecting
the appropriate loader based on file type.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

from ophthalmic_registration.core.image_data import (
    ImageData,
    ImageMetadata,
    PixelSpacing,
    ImageModality,
)
from ophthalmic_registration.core.exceptions import ImageLoadError
from ophthalmic_registration.io.dicom_loader import DicomLoader
from ophthalmic_registration.io.standard_loader import StandardImageLoader

logger = logging.getLogger(__name__)


class ImageLoader:
    """
    Unified image loader supporting DICOM and standard formats.
    
    Automatically detects the file type and uses the appropriate
    loader. Provides a consistent interface for loading ophthalmic
    images regardless of source format.
    
    Attributes:
        dicom_loader: DicomLoader instance for DICOM files
        standard_loader: StandardImageLoader instance for other formats
        default_pixel_spacing: Default pixel spacing when not available
    
    Example:
        >>> loader = ImageLoader()
        >>> 
        >>> # Load DICOM (auto-detected)
        >>> dicom_image = loader.load("fundus.dcm")
        >>> 
        >>> # Load PNG with manual pixel spacing
        >>> png_image = loader.load(
        ...     "fundus.png",
        ...     pixel_spacing=PixelSpacing(0.01, 0.01, "mm", "manual")
        ... )
        >>> 
        >>> # Load with modality specification
        >>> faf_image = loader.load(
        ...     "faf_image.tiff",
        ...     modality=ImageModality.FAF,
        ...     laterality="OD"
        ... )
    """
    
    def __init__(
        self,
        apply_dicom_luts: bool = True,
        normalize_to_uint8: bool = False,
        preserve_color: bool = True,
        default_pixel_spacing: Optional[PixelSpacing] = None
    ):
        """
        Initialize the unified image loader.
        
        Args:
            apply_dicom_luts: Apply modality/VOI LUTs for DICOM images
            normalize_to_uint8: Normalize pixel values to uint8 range
            preserve_color: Preserve color channels in standard images
            default_pixel_spacing: Default spacing when not available
        """
        self.normalize_to_uint8 = normalize_to_uint8
        self.default_pixel_spacing = default_pixel_spacing
        
        # Initialize loaders
        try:
            self.dicom_loader = DicomLoader(
                apply_modality_lut=apply_dicom_luts,
                apply_voi_lut=apply_dicom_luts,
                normalize_to_uint8=normalize_to_uint8
            )
            self._dicom_available = True
        except ImportError:
            self.dicom_loader = None
            self._dicom_available = False
            logger.warning("DICOM support not available (pydicom not installed)")
        
        self.standard_loader = StandardImageLoader(
            use_opencv=True,
            preserve_color=preserve_color
        )
    
    def load(
        self,
        filepath: Union[str, Path],
        pixel_spacing: Optional[PixelSpacing] = None,
        modality: ImageModality = ImageModality.UNKNOWN,
        laterality: Optional[str] = None,
        frame_index: int = 0,
        force_dicom: bool = False,
        force_standard: bool = False,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> ImageData:
        """
        Load an image file with automatic format detection.
        
        Args:
            filepath: Path to the image file
            pixel_spacing: Manual pixel spacing (overrides file metadata)
            modality: Imaging modality specification
            laterality: Eye laterality ('OD' or 'OS')
            frame_index: Frame index for multi-frame DICOMs
            force_dicom: Force loading as DICOM
            force_standard: Force loading as standard image
            custom_metadata: Additional custom metadata
        
        Returns:
            ImageData containing pixel array and metadata
        
        Raises:
            ImageLoadError: If the file cannot be loaded
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise ImageLoadError(str(filepath), "File does not exist")
        
        # Determine loader to use
        use_dicom = self._should_use_dicom(filepath, force_dicom, force_standard)
        
        if use_dicom:
            if not self._dicom_available:
                raise ImageLoadError(
                    str(filepath),
                    "DICOM loading requested but pydicom is not available"
                )
            image_data = self.dicom_loader.load(filepath, frame_index=frame_index)
        else:
            image_data = self.standard_loader.load(
                filepath,
                pixel_spacing=pixel_spacing,
                modality=modality,
                laterality=laterality,
                custom_metadata=custom_metadata
            )
        
        # Override pixel spacing if provided
        if pixel_spacing is not None:
            image_data.metadata.pixel_spacing = pixel_spacing
        
        # Apply default pixel spacing if still missing
        if image_data.metadata.pixel_spacing is None and self.default_pixel_spacing:
            image_data.metadata.pixel_spacing = self.default_pixel_spacing
            logger.info("Applied default pixel spacing")
        
        # Override modality if specified and not unknown
        if modality != ImageModality.UNKNOWN:
            image_data.metadata.modality = modality
        
        # Override laterality if specified
        if laterality is not None:
            image_data.metadata.laterality = laterality
        
        # Add custom metadata
        if custom_metadata:
            image_data.metadata.custom.update(custom_metadata)
        
        return image_data
    
    def _should_use_dicom(
        self,
        filepath: Path,
        force_dicom: bool,
        force_standard: bool
    ) -> bool:
        """Determine if DICOM loader should be used."""
        if force_dicom:
            return True
        if force_standard:
            return False
        
        # Check extension first
        if filepath.suffix.lower() == '.dcm':
            return True
        
        # Check if it's a known standard format
        if StandardImageLoader.is_supported_format(filepath):
            return False
        
        # Try to detect DICOM by attempting to read
        if self._dicom_available:
            try:
                if DicomLoader.is_dicom_file(filepath):
                    return True
            except Exception:
                pass
        
        # Default to standard loader
        return False
    
    def load_pair(
        self,
        baseline_path: Union[str, Path],
        followup_path: Union[str, Path],
        **kwargs
    ) -> tuple:
        """
        Load a pair of images for registration.
        
        Convenience method for loading baseline and follow-up images
        with the same settings.
        
        Args:
            baseline_path: Path to baseline image
            followup_path: Path to follow-up image
            **kwargs: Additional arguments passed to load()
        
        Returns:
            Tuple of (baseline_image, followup_image) as ImageData objects
        
        Raises:
            ImageLoadError: If either file cannot be loaded
        """
        baseline = self.load(baseline_path, **kwargs)
        followup = self.load(followup_path, **kwargs)
        
        # Log scale mismatch warning if both have pixel spacing
        if baseline.pixel_spacing and followup.pixel_spacing:
            baseline_spacing = baseline.pixel_spacing.mean_spacing
            followup_spacing = followup.pixel_spacing.mean_spacing
            
            spacing_ratio = max(baseline_spacing, followup_spacing) / min(baseline_spacing, followup_spacing)
            if spacing_ratio > 1.1:  # >10% difference
                logger.warning(
                    f"Pixel spacing mismatch detected: "
                    f"baseline={baseline_spacing:.4f}mm, "
                    f"followup={followup_spacing:.4f}mm "
                    f"(ratio={spacing_ratio:.2f})"
                )
        
        return baseline, followup
    
    def load_series(
        self,
        filepaths: List[Union[str, Path]],
        **kwargs
    ) -> List[ImageData]:
        """
        Load a series of images.
        
        Args:
            filepaths: List of image file paths
            **kwargs: Additional arguments passed to load()
        
        Returns:
            List of ImageData objects
        """
        return [self.load(fp, **kwargs) for fp in filepaths]
    
    def get_file_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about an image file without loading pixel data.
        
        Args:
            filepath: Path to the image file
        
        Returns:
            Dictionary with file format, dimensions, and available metadata
        """
        filepath = Path(filepath)
        
        info = {
            "filepath": str(filepath),
            "filename": filepath.name,
            "extension": filepath.suffix.lower(),
            "is_dicom": False,
        }
        
        # Check if DICOM
        if self._dicom_available:
            try:
                if DicomLoader.is_dicom_file(filepath):
                    info["is_dicom"] = True
                    info["frame_count"] = self.dicom_loader.get_frame_count(filepath)
            except Exception:
                pass
        
        # Get standard image info
        if not info["is_dicom"]:
            try:
                std_info = self.standard_loader.get_image_info(filepath)
                info.update(std_info)
            except Exception as e:
                info["error"] = str(e)
        
        return info
    
    @property
    def supports_dicom(self) -> bool:
        """Check if DICOM loading is available."""
        return self._dicom_available
