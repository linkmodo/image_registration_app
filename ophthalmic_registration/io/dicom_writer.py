"""
DICOM writer for saving registered ophthalmic images.

This module handles writing DICOM files while preserving original metadata
and adding registration-specific information.
"""

import logging
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import numpy as np

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

from ophthalmic_registration.core.image_data import ImageData, ImageMetadata
from ophthalmic_registration.core.exceptions import ImageLoadError

logger = logging.getLogger(__name__)


class DicomWriter:
    """
    Writer for DICOM ophthalmic images.
    
    Preserves original DICOM metadata while allowing updates for registered images.
    Adds appropriate tags to indicate image has been processed through registration.
    
    Example:
        >>> writer = DicomWriter()
        >>> writer.save(registered_image, "registered.dcm", 
        ...             source_dicom="baseline.dcm",
        ...             description="Registered to baseline")
    """
    
    def __init__(self):
        """Initialize the DICOM writer."""
        if not PYDICOM_AVAILABLE:
            raise ImportError(
                "pydicom is required for DICOM writing. "
                "Install with: pip install pydicom"
            )
    
    def save(
        self,
        image: ImageData,
        output_path: Union[str, Path],
        source_dicom: Optional[Union[str, Path]] = None,
        description: Optional[str] = None,
        add_registration_note: bool = True
    ) -> Path:
        """
        Save ImageData as DICOM file.
        
        Args:
            image: ImageData to save
            output_path: Output file path
            source_dicom: Path to source DICOM to copy metadata from
            description: Series description to add
            add_registration_note: Add note about registration processing
        
        Returns:
            Path to saved DICOM file
        """
        output_path = Path(output_path)
        
        # Load source DICOM if provided
        if source_dicom:
            try:
                ds = pydicom.dcmread(str(source_dicom))
                logger.info(f"Loaded source DICOM metadata from {source_dicom}")
            except Exception as e:
                logger.warning(f"Could not load source DICOM: {e}. Creating new DICOM.")
                ds = self._create_new_dataset()
        else:
            ds = self._create_new_dataset()
        
        # Update pixel data
        pixel_array = image.pixel_array
        
        # Ensure proper data type
        if pixel_array.dtype == np.float32 or pixel_array.dtype == np.float64:
            # Normalize to uint16 for better precision
            pixel_min, pixel_max = pixel_array.min(), pixel_array.max()
            if pixel_max > pixel_min:
                pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 65535).astype(np.uint16)
            else:
                pixel_array = np.zeros_like(pixel_array, dtype=np.uint16)
        elif pixel_array.dtype != np.uint8 and pixel_array.dtype != np.uint16:
            pixel_array = pixel_array.astype(np.uint16)
        
        ds.PixelData = pixel_array.tobytes()
        
        # Update image dimensions
        if pixel_array.ndim == 2:
            ds.Rows = pixel_array.shape[0]
            ds.Columns = pixel_array.shape[1]
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
        elif pixel_array.ndim == 3:
            ds.Rows = pixel_array.shape[0]
            ds.Columns = pixel_array.shape[1]
            ds.SamplesPerPixel = pixel_array.shape[2]
            ds.PhotometricInterpretation = "RGB"
            ds.PlanarConfiguration = 0
        
        # Update bit depth
        if pixel_array.dtype == np.uint8:
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
        elif pixel_array.dtype == np.uint16:
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
        
        ds.PixelRepresentation = 0  # Unsigned
        
        # Update metadata from ImageData if available
        if image.metadata:
            self._update_metadata_from_imagedata(ds, image.metadata)
        
        # Update UIDs to indicate this is a derived image
        ds.SOPInstanceUID = generate_uid()
        if not hasattr(ds, 'SeriesInstanceUID') or not ds.SeriesInstanceUID:
            ds.SeriesInstanceUID = generate_uid()
        
        # Add series description
        if description:
            ds.SeriesDescription = description
        elif add_registration_note:
            current_desc = getattr(ds, 'SeriesDescription', '')
            if current_desc:
                ds.SeriesDescription = f"{current_desc} - Registered"
            else:
                ds.SeriesDescription = "Registered Image"
        
        # Add image comments about registration
        if add_registration_note:
            current_comments = getattr(ds, 'ImageComments', '')
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            registration_note = f"Image registered using ophthalmic_registration pipeline. Processed: {timestamp}. "
            registration_note += "Pixel spacing preserved from reference image. "
            registration_note += "Spatial alignment applied via feature-based registration."
            
            if current_comments:
                ds.ImageComments = f"{current_comments}\n{registration_note}"
            else:
                ds.ImageComments = registration_note
        
        # Update content date/time
        now = datetime.now()
        ds.ContentDate = now.strftime("%Y%m%d")
        ds.ContentTime = now.strftime("%H%M%S")
        
        # Ensure required DICOM attributes
        if not hasattr(ds, 'StudyInstanceUID') or not ds.StudyInstanceUID:
            ds.StudyInstanceUID = generate_uid()
        if not hasattr(ds, 'SOPClassUID') or not ds.SOPClassUID:
            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.77.1.5.1"  # Ophthalmic Photography 8 Bit
        
        # Set transfer syntax
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.file_meta = self._create_file_meta(ds)
        
        # Save
        ds.save_as(str(output_path), write_like_original=False)
        logger.info(f"Saved DICOM to {output_path}")
        
        return output_path
    
    def _create_new_dataset(self) -> Dataset:
        """Create a new minimal DICOM dataset."""
        ds = Dataset()
        
        # Required DICOM attributes
        ds.PatientName = "ANONYMOUS"
        ds.PatientID = "ANON"
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = generate_uid()
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.77.1.5.1"
        ds.Modality = "OP"
        
        # Date/time
        now = datetime.now()
        ds.StudyDate = now.strftime("%Y%m%d")
        ds.SeriesDate = now.strftime("%Y%m%d")
        ds.ContentDate = now.strftime("%Y%m%d")
        ds.StudyTime = now.strftime("%H%M%S")
        ds.SeriesTime = now.strftime("%H%M%S")
        ds.ContentTime = now.strftime("%H%M%S")
        
        return ds
    
    def _update_metadata_from_imagedata(self, ds: Dataset, metadata: ImageMetadata) -> None:
        """Update DICOM dataset with metadata from ImageData."""
        # Pixel spacing - preserve from reference
        if metadata.pixel_spacing:
            ps = metadata.pixel_spacing
            ds.PixelSpacing = [ps.row_spacing, ps.column_spacing]
            # Also set ImagerPixelSpacing for fundus images
            ds.ImagerPixelSpacing = [ps.row_spacing, ps.column_spacing]
        
        # Patient/study info
        if metadata.patient_id:
            ds.PatientID = metadata.patient_id
        if metadata.study_uid:
            ds.StudyInstanceUID = metadata.study_uid
        if metadata.laterality:
            ds.ImageLaterality = metadata.laterality
            ds.Laterality = metadata.laterality
        
        # Equipment info
        if metadata.manufacturer:
            ds.Manufacturer = metadata.manufacturer
        if metadata.model_name:
            ds.ManufacturerModelName = metadata.model_name
        
        # Acquisition date
        if metadata.acquisition_date:
            ds.AcquisitionDate = metadata.acquisition_date.strftime("%Y%m%d")
        
        # Image dimensions
        if metadata.rows:
            ds.Rows = metadata.rows
        if metadata.columns:
            ds.Columns = metadata.columns
    
    def _create_file_meta(self, ds: Dataset) -> Dataset:
        """Create DICOM file meta information."""
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.ImplementationVersionName = "OPHTHALMIC_REG_1.0"
        return file_meta
