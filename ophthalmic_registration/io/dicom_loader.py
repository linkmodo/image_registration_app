"""
DICOM image loader for ophthalmic images.

This module handles loading and parsing of DICOM files, including extraction
of pixel data and relevant metadata for ophthalmic imaging modalities.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Union
from datetime import datetime
import numpy as np

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

from ophthalmic_registration.core.image_data import (
    ImageData,
    ImageMetadata,
    PixelSpacing,
    ImageModality,
)
from ophthalmic_registration.core.exceptions import ImageLoadError, MetadataError

logger = logging.getLogger(__name__)


class DicomLoader:
    """
    Loader for DICOM ophthalmic images.
    
    Handles both single-frame and multi-frame DICOM files, extracts
    relevant metadata, and normalizes pixel data for processing.
    
    Attributes:
        apply_modality_lut: Whether to apply modality LUT transformation
        apply_voi_lut: Whether to apply VOI LUT transformation
        normalize_to_uint8: Whether to normalize output to uint8 range
    
    Example:
        >>> loader = DicomLoader()
        >>> image_data = loader.load("fundus_image.dcm")
        >>> print(image_data.metadata.pixel_spacing)
    """
    
    # Mapping of DICOM modality codes to ImageModality
    MODALITY_MAP = {
        "OP": ImageModality.FUNDUS,
        "OPT": ImageModality.OCT_ENFACE,
        "XC": ImageModality.COLOR,
    }
    
    # Ophthalmic-specific SOP Class UIDs
    OPHTHALMIC_SOP_CLASSES = {
        "1.2.840.10008.5.1.4.1.1.77.1.5.1": ImageModality.FUNDUS,  # Ophthalmic Photography 8 Bit
        "1.2.840.10008.5.1.4.1.1.77.1.5.2": ImageModality.FUNDUS,  # Ophthalmic Photography 16 Bit
        "1.2.840.10008.5.1.4.1.1.77.1.5.4": ImageModality.OCT_ENFACE,  # Ophthalmic Tomography
    }
    
    def __init__(
        self,
        apply_modality_lut: bool = True,
        apply_voi_lut: bool = True,
        normalize_to_uint8: bool = False
    ):
        """
        Initialize the DICOM loader.
        
        Args:
            apply_modality_lut: Apply modality LUT if available
            apply_voi_lut: Apply VOI LUT (window/level) if available
            normalize_to_uint8: Normalize output to uint8 range
        
        Raises:
            ImportError: If pydicom is not installed
        """
        if not PYDICOM_AVAILABLE:
            raise ImportError(
                "pydicom is required for DICOM loading. "
                "Install with: pip install pydicom"
            )
        
        self.apply_modality_lut = apply_modality_lut
        self.apply_voi_lut = apply_voi_lut
        self.normalize_to_uint8 = normalize_to_uint8
    
    def load(
        self,
        filepath: Union[str, Path],
        frame_index: int = 0
    ) -> ImageData:
        """
        Load a DICOM image file.
        
        Args:
            filepath: Path to the DICOM file
            frame_index: Frame index for multi-frame DICOMs (0-indexed)
        
        Returns:
            ImageData containing pixel array and extracted metadata
        
        Raises:
            ImageLoadError: If file cannot be loaded or parsed
            MetadataError: If critical metadata is missing or invalid
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise ImageLoadError(str(filepath), "File does not exist")
        
        try:
            ds = pydicom.dcmread(str(filepath))
        except Exception as e:
            raise ImageLoadError(str(filepath), f"Failed to parse DICOM: {e}")
        
        # Extract pixel array
        pixel_array = self._extract_pixel_array(ds, frame_index, filepath)
        
        # Extract metadata
        metadata = self._extract_metadata(ds)
        
        # Apply LUT transformations if requested
        pixel_array = self._apply_luts(ds, pixel_array)
        
        # Normalize if requested
        if self.normalize_to_uint8:
            pixel_array = self._normalize_to_uint8(pixel_array)
        
        logger.info(
            f"Loaded DICOM: {filepath.name}, "
            f"shape={pixel_array.shape}, dtype={pixel_array.dtype}, "
            f"modality={metadata.modality.value}"
        )
        
        return ImageData(
            pixel_array=pixel_array,
            metadata=metadata,
            filepath=str(filepath)
        )
    
    def _extract_pixel_array(
        self,
        ds: 'pydicom.Dataset',
        frame_index: int,
        filepath: Path
    ) -> np.ndarray:
        """
        Extract pixel array from DICOM dataset.
        
        Handles both single-frame and multi-frame DICOMs.
        """
        try:
            pixel_array = ds.pixel_array
        except Exception as e:
            raise ImageLoadError(
                str(filepath),
                f"Failed to extract pixel data: {e}",
                details="Check if transfer syntax is supported and pixel data exists"
            )
        
        # Handle multi-frame DICOM
        if pixel_array.ndim == 3 and not self._is_color_image(ds):
            # Multi-frame grayscale
            num_frames = pixel_array.shape[0]
            if frame_index >= num_frames:
                raise ImageLoadError(
                    str(filepath),
                    f"Frame index {frame_index} out of range (total frames: {num_frames})"
                )
            pixel_array = pixel_array[frame_index]
            logger.debug(f"Extracted frame {frame_index} from multi-frame DICOM")
        
        elif pixel_array.ndim == 4:
            # Multi-frame color
            num_frames = pixel_array.shape[0]
            if frame_index >= num_frames:
                raise ImageLoadError(
                    str(filepath),
                    f"Frame index {frame_index} out of range (total frames: {num_frames})"
                )
            pixel_array = pixel_array[frame_index]
            logger.debug(f"Extracted color frame {frame_index} from multi-frame DICOM")
        
        return pixel_array
    
    def _is_color_image(self, ds: 'pydicom.Dataset') -> bool:
        """Check if the DICOM contains a color image."""
        photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
        return photometric in ('RGB', 'YBR_FULL', 'YBR_FULL_422', 'PALETTE COLOR')
    
    def _extract_metadata(self, ds: 'pydicom.Dataset') -> ImageMetadata:
        """
        Extract relevant metadata from DICOM dataset.
        
        Extracts pixel spacing, modality information, patient/study identifiers,
        and other relevant attributes for ophthalmic imaging.
        """
        metadata = ImageMetadata()
        
        # Pixel spacing
        metadata.pixel_spacing = self._extract_pixel_spacing(ds)
        
        # Image dimensions
        metadata.rows = getattr(ds, 'Rows', None)
        metadata.columns = getattr(ds, 'Columns', None)
        
        # Modality
        metadata.modality = self._determine_modality(ds)
        
        # Laterality (OD/OS)
        laterality = getattr(ds, 'ImageLaterality', None)
        if laterality is None:
            laterality = getattr(ds, 'Laterality', None)
        metadata.laterality = laterality
        
        # Acquisition date
        acq_date = getattr(ds, 'AcquisitionDate', None)
        if acq_date is None:
            acq_date = getattr(ds, 'ContentDate', None)
        if acq_date:
            try:
                metadata.acquisition_date = datetime.strptime(acq_date, '%Y%m%d')
            except ValueError:
                logger.warning(f"Could not parse acquisition date: {acq_date}")
        
        # UIDs
        metadata.patient_id = getattr(ds, 'PatientID', None)
        metadata.study_uid = getattr(ds, 'StudyInstanceUID', None)
        metadata.series_uid = getattr(ds, 'SeriesInstanceUID', None)
        metadata.instance_uid = getattr(ds, 'SOPInstanceUID', None)
        
        # Equipment info
        metadata.manufacturer = getattr(ds, 'Manufacturer', None)
        metadata.model_name = getattr(ds, 'ManufacturerModelName', None)
        
        # Bit depth
        metadata.bits_stored = getattr(ds, 'BitsStored', None)
        metadata.bits_allocated = getattr(ds, 'BitsAllocated', None)
        
        # Photometric interpretation
        metadata.photometric_interpretation = getattr(ds, 'PhotometricInterpretation', None)
        
        # Slice thickness (for OCT)
        metadata.slice_thickness = getattr(ds, 'SliceThickness', None)
        
        # Image orientation
        orientation = getattr(ds, 'ImageOrientationPatient', None)
        if orientation:
            metadata.image_orientation = [float(x) for x in orientation]
        
        # Window/level
        metadata.window_center = self._get_window_value(ds, 'WindowCenter')
        metadata.window_width = self._get_window_value(ds, 'WindowWidth')
        
        return metadata
    
    def _extract_pixel_spacing(self, ds: 'pydicom.Dataset') -> Optional[PixelSpacing]:
        """
        Extract pixel spacing from DICOM dataset.
        
        Checks multiple DICOM tags that may contain spatial calibration:
        - PixelSpacing
        - ImagerPixelSpacing
        - SharedFunctionalGroupsSequence (for enhanced DICOM)
        - PixelAspectRatio (for relative spacing)
        """
        # Try standard PixelSpacing tag
        pixel_spacing = getattr(ds, 'PixelSpacing', None)
        if pixel_spacing:
            return PixelSpacing(
                row_spacing=float(pixel_spacing[0]),
                column_spacing=float(pixel_spacing[1]),
                unit="mm",
                source="dicom"
            )
        
        # Try ImagerPixelSpacing (common in fundus imaging)
        imager_spacing = getattr(ds, 'ImagerPixelSpacing', None)
        if imager_spacing:
            return PixelSpacing(
                row_spacing=float(imager_spacing[0]),
                column_spacing=float(imager_spacing[1]),
                unit="mm",
                source="dicom"
            )
        
        # Try enhanced DICOM functional groups
        shared_fg = getattr(ds, 'SharedFunctionalGroupsSequence', None)
        if shared_fg:
            for fg in shared_fg:
                pixel_measures = getattr(fg, 'PixelMeasuresSequence', None)
                if pixel_measures:
                    for pm in pixel_measures:
                        ps = getattr(pm, 'PixelSpacing', None)
                        if ps:
                            return PixelSpacing(
                                row_spacing=float(ps[0]),
                                column_spacing=float(ps[1]),
                                unit="mm",
                                source="dicom"
                            )
        
        # Check for ophthalmic-specific tags
        # Acquisition Device Type Sequence may contain FOV information
        acq_device_seq = getattr(ds, 'AcquisitionDeviceTypeCodeSequence', None)
        if acq_device_seq:
            logger.debug("Found AcquisitionDeviceTypeCodeSequence but no pixel spacing")
        
        logger.warning("No pixel spacing information found in DICOM")
        return None
    
    def _determine_modality(self, ds: 'pydicom.Dataset') -> ImageModality:
        """
        Determine the ophthalmic imaging modality.
        
        Uses SOP Class UID, Modality tag, and other attributes to
        determine the specific imaging modality.
        """
        # Check SOP Class UID first (most specific)
        sop_class = getattr(ds, 'SOPClassUID', None)
        if sop_class and str(sop_class) in self.OPHTHALMIC_SOP_CLASSES:
            return self.OPHTHALMIC_SOP_CLASSES[str(sop_class)]
        
        # Check Modality tag
        modality = getattr(ds, 'Modality', None)
        if modality and modality in self.MODALITY_MAP:
            return self.MODALITY_MAP[modality]
        
        # Try to infer from Series Description or other attributes
        series_desc = getattr(ds, 'SeriesDescription', '') or ''
        series_desc = series_desc.lower()
        
        if 'faf' in series_desc or 'autofluorescence' in series_desc:
            return ImageModality.FAF
        elif 'ir' in series_desc or 'infrared' in series_desc:
            return ImageModality.IR_ENFACE
        elif 'icg' in series_desc or 'indocyanine' in series_desc:
            return ImageModality.ICGA
        elif 'fa' in series_desc or 'fluorescein' in series_desc:
            return ImageModality.FA
        elif 'fundus' in series_desc or 'color' in series_desc:
            return ImageModality.FUNDUS
        elif 'red' in series_desc and 'free' in series_desc:
            return ImageModality.RED_FREE
        
        return ImageModality.UNKNOWN
    
    def _get_window_value(
        self,
        ds: 'pydicom.Dataset',
        tag_name: str
    ) -> Optional[float]:
        """Extract window center or width, handling multi-valued cases."""
        value = getattr(ds, tag_name, None)
        if value is None:
            return None
        
        # May be a list for multi-frame or multi-window
        if hasattr(value, '__iter__') and not isinstance(value, str):
            try:
                return float(value[0])
            except (IndexError, TypeError, ValueError):
                return None
        
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    
    def _apply_luts(
        self,
        ds: 'pydicom.Dataset',
        pixel_array: np.ndarray
    ) -> np.ndarray:
        """Apply modality and VOI LUT transformations."""
        result = pixel_array.copy()
        
        if self.apply_modality_lut:
            try:
                result = apply_modality_lut(result, ds)
            except Exception as e:
                logger.debug(f"Could not apply modality LUT: {e}")
        
        if self.apply_voi_lut:
            try:
                result = apply_voi_lut(result, ds, index=0)
            except Exception as e:
                logger.debug(f"Could not apply VOI LUT: {e}")
        
        return result
    
    def _normalize_to_uint8(self, pixel_array: np.ndarray) -> np.ndarray:
        """Normalize pixel array to uint8 range [0, 255]."""
        arr = pixel_array.astype(np.float64)
        arr_min, arr_max = arr.min(), arr.max()
        
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr, dtype=np.float64)
        
        return arr.astype(np.uint8)
    
    def get_frame_count(self, filepath: Union[str, Path]) -> int:
        """
        Get the number of frames in a DICOM file.
        
        Args:
            filepath: Path to the DICOM file
        
        Returns:
            Number of frames (1 for single-frame DICOMs)
        """
        filepath = Path(filepath)
        ds = pydicom.dcmread(str(filepath), stop_before_pixels=True)
        
        num_frames = getattr(ds, 'NumberOfFrames', None)
        if num_frames:
            return int(num_frames)
        
        return 1
    
    def load_all_frames(self, filepath: Union[str, Path]) -> List[ImageData]:
        """
        Load all frames from a multi-frame DICOM.
        
        Args:
            filepath: Path to the DICOM file
        
        Returns:
            List of ImageData objects, one per frame
        """
        num_frames = self.get_frame_count(filepath)
        return [self.load(filepath, frame_index=i) for i in range(num_frames)]
    
    @staticmethod
    def is_dicom_file(filepath: Union[str, Path]) -> bool:
        """
        Check if a file is a valid DICOM file.
        
        Args:
            filepath: Path to check
        
        Returns:
            True if the file appears to be a valid DICOM
        """
        if not PYDICOM_AVAILABLE:
            return False
        
        filepath = Path(filepath)
        if not filepath.exists():
            return False
        
        try:
            pydicom.dcmread(str(filepath), stop_before_pixels=True, force=False)
            return True
        except Exception:
            return False
