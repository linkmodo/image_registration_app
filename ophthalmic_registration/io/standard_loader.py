"""
Standard image format loader for ophthalmic images.

This module handles loading of standard image formats (PNG, TIFF, JPEG, BMP)
using OpenCV and Pillow, with optional manual metadata specification.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from ophthalmic_registration.core.image_data import (
    ImageData,
    ImageMetadata,
    PixelSpacing,
    ImageModality,
)
from ophthalmic_registration.core.exceptions import ImageLoadError

logger = logging.getLogger(__name__)


class StandardImageLoader:
    """
    Loader for standard image formats (PNG, TIFF, JPEG, BMP).
    
    Supports loading via OpenCV or Pillow, with optional manual
    specification of pixel spacing and other metadata.
    
    Attributes:
        use_opencv: Prefer OpenCV for loading (faster)
        preserve_color: Keep color images as-is (don't convert to grayscale)
    
    Example:
        >>> loader = StandardImageLoader()
        >>> image_data = loader.load(
        ...     "fundus.png",
        ...     pixel_spacing=PixelSpacing(0.01, 0.01, "mm", "manual")
        ... )
    """
    
    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}
    
    def __init__(
        self,
        use_opencv: bool = True,
        preserve_color: bool = True
    ):
        """
        Initialize the standard image loader.
        
        Args:
            use_opencv: Prefer OpenCV for loading (default True)
            preserve_color: Preserve color channels (default True)
        
        Raises:
            ImportError: If neither OpenCV nor Pillow is available
        """
        self.use_opencv = use_opencv and CV2_AVAILABLE
        self.preserve_color = preserve_color
        
        if not CV2_AVAILABLE and not PIL_AVAILABLE:
            raise ImportError(
                "Either opencv-python or Pillow is required for standard image loading. "
                "Install with: pip install opencv-python Pillow"
            )
        
        if use_opencv and not CV2_AVAILABLE:
            logger.warning("OpenCV not available, falling back to Pillow")
            self.use_opencv = False
    
    def load(
        self,
        filepath: Union[str, Path],
        pixel_spacing: Optional[PixelSpacing] = None,
        modality: ImageModality = ImageModality.UNKNOWN,
        laterality: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> ImageData:
        """
        Load a standard image file.
        
        Args:
            filepath: Path to the image file
            pixel_spacing: Optional manual pixel spacing specification
            modality: Optional imaging modality specification
            laterality: Optional eye laterality ('OD' or 'OS')
            custom_metadata: Optional dictionary of custom metadata
        
        Returns:
            ImageData containing pixel array and metadata
        
        Raises:
            ImageLoadError: If the file cannot be loaded
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise ImageLoadError(str(filepath), "File does not exist")
        
        if filepath.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ImageLoadError(
                str(filepath),
                f"Unsupported format: {filepath.suffix}",
                details=f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        
        # Load pixel data
        if self.use_opencv:
            pixel_array = self._load_with_opencv(filepath)
        else:
            pixel_array = self._load_with_pillow(filepath)
        
        # Build metadata
        metadata = self._build_metadata(
            filepath=filepath,
            pixel_array=pixel_array,
            pixel_spacing=pixel_spacing,
            modality=modality,
            laterality=laterality,
            custom_metadata=custom_metadata
        )
        
        logger.info(
            f"Loaded image: {filepath.name}, "
            f"shape={pixel_array.shape}, dtype={pixel_array.dtype}"
        )
        
        return ImageData(
            pixel_array=pixel_array,
            metadata=metadata,
            filepath=str(filepath)
        )
    
    def _load_with_opencv(self, filepath: Path) -> np.ndarray:
        """Load image using OpenCV."""
        if self.preserve_color:
            # Load as-is, preserving color
            pixel_array = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        else:
            # Load as grayscale
            pixel_array = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        
        if pixel_array is None:
            raise ImageLoadError(str(filepath), "OpenCV failed to load image")
        
        # Convert BGR to RGB for color images
        if pixel_array.ndim == 3 and pixel_array.shape[2] == 3:
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB)
        elif pixel_array.ndim == 3 and pixel_array.shape[2] == 4:
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_BGRA2RGBA)
        
        return pixel_array
    
    def _load_with_pillow(self, filepath: Path) -> np.ndarray:
        """Load image using Pillow."""
        try:
            img = Image.open(filepath)
            
            if not self.preserve_color and img.mode not in ('L', 'I', 'F'):
                img = img.convert('L')
            
            pixel_array = np.array(img)
            
            return pixel_array
        except Exception as e:
            raise ImageLoadError(str(filepath), f"Pillow failed to load image: {e}")
    
    def _build_metadata(
        self,
        filepath: Path,
        pixel_array: np.ndarray,
        pixel_spacing: Optional[PixelSpacing],
        modality: ImageModality,
        laterality: Optional[str],
        custom_metadata: Optional[Dict[str, Any]]
    ) -> ImageMetadata:
        """Build metadata object for the loaded image."""
        metadata = ImageMetadata()
        
        # Image dimensions
        metadata.rows = pixel_array.shape[0]
        metadata.columns = pixel_array.shape[1]
        
        # Pixel spacing (manual or from TIFF)
        if pixel_spacing:
            metadata.pixel_spacing = pixel_spacing
        elif filepath.suffix.lower() in ('.tiff', '.tif'):
            # Try to extract from TIFF metadata
            tiff_spacing = self._extract_tiff_spacing(filepath)
            if tiff_spacing:
                metadata.pixel_spacing = tiff_spacing
        
        # Modality
        metadata.modality = modality
        
        # Laterality
        metadata.laterality = laterality
        
        # Bit depth
        if pixel_array.dtype == np.uint8:
            metadata.bits_stored = 8
            metadata.bits_allocated = 8
        elif pixel_array.dtype == np.uint16:
            metadata.bits_stored = 16
            metadata.bits_allocated = 16
        
        # Photometric interpretation
        if pixel_array.ndim == 2:
            metadata.photometric_interpretation = "MONOCHROME2"
        elif pixel_array.ndim == 3:
            if pixel_array.shape[2] == 3:
                metadata.photometric_interpretation = "RGB"
            elif pixel_array.shape[2] == 4:
                metadata.photometric_interpretation = "RGBA"
        
        # Custom metadata
        if custom_metadata:
            metadata.custom = custom_metadata
        
        return metadata
    
    def _extract_tiff_spacing(self, filepath: Path) -> Optional[PixelSpacing]:
        """
        Extract pixel spacing from TIFF metadata.
        
        Checks XResolution/YResolution tags and ResolutionUnit.
        """
        if not PIL_AVAILABLE:
            return None
        
        try:
            img = Image.open(filepath)
            
            # Get TIFF tags
            x_res = img.tag_v2.get(282)  # XResolution
            y_res = img.tag_v2.get(283)  # YResolution
            res_unit = img.tag_v2.get(296, 2)  # ResolutionUnit (default: inch)
            
            if x_res is None or y_res is None:
                return None
            
            # Convert resolution to spacing
            # Resolution is pixels per unit, spacing is units per pixel
            x_spacing = 1.0 / float(x_res)
            y_spacing = 1.0 / float(y_res)
            
            # Convert to mm based on ResolutionUnit
            # 1 = No unit, 2 = inch, 3 = centimeter
            if res_unit == 2:  # inch
                x_spacing *= 25.4  # inches to mm
                y_spacing *= 25.4
            elif res_unit == 3:  # centimeter
                x_spacing *= 10.0  # cm to mm
                y_spacing *= 10.0
            else:
                # Unknown unit, return as-is with warning
                logger.warning(f"Unknown TIFF resolution unit: {res_unit}")
                return None
            
            return PixelSpacing(
                row_spacing=y_spacing,
                column_spacing=x_spacing,
                unit="mm",
                source="tiff"
            )
        except Exception as e:
            logger.debug(f"Could not extract TIFF spacing: {e}")
            return None
    
    def get_image_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get basic image information without loading full pixel data.
        
        Args:
            filepath: Path to the image file
        
        Returns:
            Dictionary with image dimensions, format, and available metadata
        """
        filepath = Path(filepath)
        
        info = {
            "filepath": str(filepath),
            "filename": filepath.name,
            "extension": filepath.suffix.lower(),
            "file_size_bytes": filepath.stat().st_size if filepath.exists() else 0,
        }
        
        if PIL_AVAILABLE:
            try:
                img = Image.open(filepath)
                info.update({
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format,
                })
                
                # Extract EXIF if available
                exif = img.getexif()
                if exif:
                    info["exif"] = {
                        TAGS.get(k, k): v
                        for k, v in exif.items()
                        if isinstance(v, (str, int, float))
                    }
            except Exception as e:
                logger.debug(f"Could not get PIL image info: {e}")
        
        return info
    
    @staticmethod
    def is_supported_format(filepath: Union[str, Path]) -> bool:
        """
        Check if a file has a supported image format extension.
        
        Args:
            filepath: Path to check
        
        Returns:
            True if the file extension is supported
        """
        filepath = Path(filepath)
        return filepath.suffix.lower() in StandardImageLoader.SUPPORTED_EXTENSIONS
