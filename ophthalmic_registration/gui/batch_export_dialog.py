"""
Batch export dialog for registered images.

Provides options for exporting multiple aligned follow-up images
with support for various formats including DICOM, lossless (PNG, TIFF),
and lossy (JPEG with quality settings).
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QSpinBox, QCheckBox, QPushButton, QProgressBar,
    QFileDialog, QListWidget, QListWidgetItem, QLineEdit,
    QMessageBox, QFrame, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread

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
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

from ophthalmic_registration.core.image_data import ImageData, TransformResult

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Available export formats."""
    PNG = "png"
    TIFF = "tiff"
    JPEG = "jpeg"
    DICOM = "dicom"


@dataclass
class ExportSettings:
    """Export settings configuration."""
    format: ExportFormat = ExportFormat.PNG
    jpeg_quality: int = 95
    include_baseline: bool = False
    include_overlay: bool = False
    include_difference: bool = False
    include_transform: bool = True
    include_report: bool = True
    preserve_original_name: bool = True
    output_suffix: str = "_registered"


@dataclass
class RegisteredImage:
    """Container for a registered image and its metadata."""
    original_path: str
    original_image: ImageData
    registered_image: ImageData
    transform_result: TransformResult
    visit_name: str = ""


class ExportWorker(QThread):
    """Worker thread for batch export."""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(int, int)  # success_count, fail_count
    error = pyqtSignal(str)
    
    def __init__(
        self,
        images: List[RegisteredImage],
        baseline: ImageData,
        output_dir: Path,
        settings: ExportSettings
    ):
        super().__init__()
        self.images = images
        self.baseline = baseline
        self.output_dir = output_dir
        self.settings = settings
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def run(self):
        success_count = 0
        fail_count = 0
        total = len(self.images)
        
        # Export baseline if requested
        if self.settings.include_baseline:
            try:
                self._export_image(
                    self.baseline,
                    "baseline",
                    None
                )
            except Exception as e:
                logger.error(f"Failed to export baseline: {e}")
        
        for i, reg_image in enumerate(self.images):
            if self._cancelled:
                break
            
            progress_pct = int((i / total) * 100)
            self.progress.emit(progress_pct, f"Exporting {i+1}/{total}: {reg_image.visit_name}")
            
            try:
                self._export_registered_image(reg_image)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to export {reg_image.original_path}: {e}")
                fail_count += 1
        
        self.progress.emit(100, "Export complete")
        self.finished.emit(success_count, fail_count)
    
    def _export_registered_image(self, reg_image: RegisteredImage) -> None:
        """Export a single registered image with all requested outputs."""
        # Determine output filename
        if self.settings.preserve_original_name:
            original_name = Path(reg_image.original_path).stem
            base_name = f"{original_name}{self.settings.output_suffix}"
        else:
            base_name = f"{reg_image.visit_name}{self.settings.output_suffix}"
        
        # Export registered image
        self._export_image(
            reg_image.registered_image,
            base_name,
            reg_image.original_image if self.settings.format == ExportFormat.DICOM else None
        )
        
        # Export overlay if requested
        if self.settings.include_overlay:
            overlay = self._create_overlay(self.baseline, reg_image.registered_image)
            overlay_path = self.output_dir / f"{base_name}_overlay.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Export difference if requested
        if self.settings.include_difference:
            diff = self._create_difference(self.baseline, reg_image.registered_image)
            diff_path = self.output_dir / f"{base_name}_difference.png"
            cv2.imwrite(str(diff_path), diff)
        
        # Export transform JSON if requested
        if self.settings.include_transform:
            self._export_transform(reg_image.transform_result, f"{base_name}_transform.json")
        
        # Export report if requested
        if self.settings.include_report:
            self._export_report(reg_image.transform_result, f"{base_name}_report.txt")
    
    def _export_image(
        self,
        image: ImageData,
        base_name: str,
        original_dicom: Optional[ImageData] = None
    ) -> Path:
        """Export image in the configured format."""
        img = image.as_uint8()
        
        if self.settings.format == ExportFormat.PNG:
            output_path = self.output_dir / f"{base_name}.png"
            if CV2_AVAILABLE:
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), img)
            elif PIL_AVAILABLE:
                Image.fromarray(img).save(output_path)
                
        elif self.settings.format == ExportFormat.TIFF:
            output_path = self.output_dir / f"{base_name}.tiff"
            if CV2_AVAILABLE:
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), img)
            elif PIL_AVAILABLE:
                Image.fromarray(img).save(output_path, compression="tiff_lzw")
                
        elif self.settings.format == ExportFormat.JPEG:
            output_path = self.output_dir / f"{base_name}.jpg"
            if CV2_AVAILABLE:
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(output_path), img,
                    [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality]
                )
            elif PIL_AVAILABLE:
                pil_img = Image.fromarray(img)
                if pil_img.mode == 'RGBA':
                    pil_img = pil_img.convert('RGB')
                pil_img.save(output_path, quality=self.settings.jpeg_quality)
                
        elif self.settings.format == ExportFormat.DICOM:
            output_path = self.output_dir / f"{base_name}.dcm"
            self._export_dicom(image, output_path, original_dicom)
        
        logger.info(f"Exported: {output_path}")
        return output_path
    
    def _export_dicom(
        self,
        image: ImageData,
        output_path: Path,
        original_dicom: Optional[ImageData] = None
    ) -> None:
        """Export image as DICOM file."""
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom required for DICOM export")
        
        img = image.as_uint8()
        
        # Convert to grayscale if needed for standard DICOM
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Create DICOM dataset
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.5.1'  # Ophthalmic Photography
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        
        ds = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Copy metadata from original if available
        if original_dicom is not None and hasattr(original_dicom, 'metadata'):
            meta = original_dicom.metadata
            if hasattr(meta, 'patient_id') and meta.patient_id:
                ds.PatientID = meta.patient_id
            if hasattr(meta, 'patient_name') and meta.patient_name:
                ds.PatientName = meta.patient_name
            if hasattr(meta, 'study_date') and meta.study_date:
                ds.StudyDate = meta.study_date
            if hasattr(meta, 'modality') and meta.modality:
                ds.Modality = meta.modality
            if hasattr(meta, 'laterality') and meta.laterality:
                ds.ImageLaterality = meta.laterality
        else:
            ds.PatientID = "UNKNOWN"
            ds.PatientName = "UNKNOWN"
            ds.Modality = "OP"  # Ophthalmic Photography
        
        # Set required DICOM attributes
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        
        # Image attributes
        ds.Rows = img.shape[0]
        ds.Columns = img.shape[1]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = img.tobytes()
        
        # Add registration note
        ds.ImageComments = "Registered image - aligned to baseline"
        
        ds.save_as(str(output_path))
    
    def _export_transform(self, result: TransformResult, filename: str) -> None:
        """Export transform to JSON."""
        import json
        from datetime import datetime
        
        output_path = self.output_dir / filename
        
        data = {
            "transform_matrix": result.transform_matrix.tolist(),
            "motion_model": result.motion_model.value,
            "translation": list(result.translation),
            "rotation_degrees": result.rotation_degrees,
            "scale_factors": list(result.scale_factors),
            "quality_metrics": result.quality_metrics,
            "registration_time_ms": result.registration_time_ms,
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_report(self, result: TransformResult, filename: str) -> None:
        """Export human-readable report."""
        from datetime import datetime
        
        output_path = self.output_dir / filename
        
        lines = [
            "=" * 50,
            "REGISTRATION REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Motion Model: {result.motion_model.value}",
            f"Processing Time: {result.registration_time_ms:.1f} ms",
            "",
            "Transform:",
            f"  Translation: ({result.translation[0]:.2f}, {result.translation[1]:.2f}) px",
            f"  Rotation: {result.rotation_degrees:.3f}°" if result.rotation_degrees else "",
            f"  Scale: ({result.scale_factors[0]:.4f}, {result.scale_factors[1]:.4f})",
            "",
            "Quality Metrics:",
        ]
        
        for key, value in result.quality_metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        
        if result.warnings:
            lines.extend(["", "Warnings:"])
            for w in result.warnings:
                lines.append(f"  - {w}")
        
        lines.extend(["", "=" * 50])
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
    
    def _create_overlay(self, img1: ImageData, img2: ImageData, alpha: float = 0.5) -> np.ndarray:
        """Create blended overlay."""
        i1 = img1.as_uint8()
        i2 = img2.as_uint8()
        
        h = min(i1.shape[0], i2.shape[0])
        w = min(i1.shape[1], i2.shape[1])
        i1 = i1[:h, :w]
        i2 = i2[:h, :w]
        
        if i1.ndim == 2:
            i1 = np.stack([i1, np.zeros_like(i1), i1], axis=-1)
        if i2.ndim == 2:
            i2 = np.stack([np.zeros_like(i2), i2, np.zeros_like(i2)], axis=-1)
        
        return cv2.addWeighted(i1, 1 - alpha, i2, alpha, 0)
    
    def _create_difference(self, img1: ImageData, img2: ImageData) -> np.ndarray:
        """Create difference map."""
        i1 = img1.as_uint8()
        i2 = img2.as_uint8()
        
        h = min(i1.shape[0], i2.shape[0])
        w = min(i1.shape[1], i2.shape[1])
        i1 = i1[:h, :w]
        i2 = i2[:h, :w]
        
        if i1.ndim == 3:
            i1 = cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY)
        if i2.ndim == 3:
            i2 = cv2.cvtColor(i2, cv2.COLOR_RGB2GRAY)
        
        return cv2.absdiff(i1, i2)


class BatchExportDialog(QDialog):
    """
    Dialog for batch exporting registered images.
    
    Provides options for:
    - Export format (PNG, TIFF, JPEG, DICOM)
    - JPEG quality settings
    - Additional outputs (overlay, difference, transform, report)
    - Output directory selection
    """
    
    def __init__(
        self,
        registered_images: List[RegisteredImage],
        baseline: ImageData,
        parent=None
    ):
        super().__init__(parent)
        self.registered_images = registered_images
        self.baseline = baseline
        self.settings = ExportSettings()
        self._worker = None
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        self.setWindowTitle("Batch Export Registered Images")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Images to export
        images_group = QGroupBox(f"Images to Export ({len(self.registered_images)} registered)")
        images_layout = QVBoxLayout(images_group)
        
        self._image_list = QListWidget()
        for img in self.registered_images:
            item = QListWidgetItem(f"✓ {Path(img.original_path).name}")
            item.setCheckState(Qt.CheckState.Checked)
            self._image_list.addItem(item)
        self._image_list.setMaximumHeight(120)
        images_layout.addWidget(self._image_list)
        
        layout.addWidget(images_group)
        
        # Format options
        format_group = QGroupBox("Export Format")
        format_layout = QVBoxLayout(format_group)
        
        self._format_group = QButtonGroup(self)
        
        # PNG (lossless)
        png_radio = QRadioButton("PNG (Lossless, recommended)")
        png_radio.setToolTip("Lossless compression, best for archival and further processing")
        png_radio.setChecked(True)
        self._format_group.addButton(png_radio, 0)
        format_layout.addWidget(png_radio)
        
        # TIFF (lossless)
        tiff_radio = QRadioButton("TIFF (Lossless, large files)")
        tiff_radio.setToolTip("Lossless compression, widely compatible with medical software")
        self._format_group.addButton(tiff_radio, 1)
        format_layout.addWidget(tiff_radio)
        
        # JPEG (lossy)
        jpeg_layout = QHBoxLayout()
        jpeg_radio = QRadioButton("JPEG (Lossy, smaller files)")
        jpeg_radio.setToolTip("Lossy compression, good for sharing but not recommended for analysis")
        self._format_group.addButton(jpeg_radio, 2)
        jpeg_layout.addWidget(jpeg_radio)
        
        jpeg_layout.addWidget(QLabel("Quality:"))
        self._jpeg_quality = QSpinBox()
        self._jpeg_quality.setRange(1, 100)
        self._jpeg_quality.setValue(95)
        self._jpeg_quality.setToolTip("JPEG quality (1-100). Higher = better quality, larger file")
        self._jpeg_quality.setEnabled(False)
        jpeg_layout.addWidget(self._jpeg_quality)
        jpeg_layout.addStretch()
        format_layout.addLayout(jpeg_layout)
        
        # DICOM
        dicom_radio = QRadioButton("DICOM (Medical format)")
        dicom_radio.setToolTip("Standard medical imaging format, preserves patient metadata")
        dicom_radio.setEnabled(PYDICOM_AVAILABLE)
        if not PYDICOM_AVAILABLE:
            dicom_radio.setText("DICOM (Not available - install pydicom)")
        self._format_group.addButton(dicom_radio, 3)
        format_layout.addWidget(dicom_radio)
        
        layout.addWidget(format_group)
        
        # Additional outputs
        outputs_group = QGroupBox("Additional Outputs")
        outputs_layout = QVBoxLayout(outputs_group)
        
        self._include_baseline = QCheckBox("Include baseline image")
        self._include_baseline.setToolTip("Export the reference baseline image as well")
        outputs_layout.addWidget(self._include_baseline)
        
        self._include_overlay = QCheckBox("Include overlay images")
        self._include_overlay.setToolTip("Create blended overlay of baseline and registered image")
        outputs_layout.addWidget(self._include_overlay)
        
        self._include_difference = QCheckBox("Include difference maps")
        self._include_difference.setToolTip("Create difference map highlighting changes")
        outputs_layout.addWidget(self._include_difference)
        
        self._include_transform = QCheckBox("Include transform JSON")
        self._include_transform.setChecked(True)
        self._include_transform.setToolTip("Save registration transform parameters")
        outputs_layout.addWidget(self._include_transform)
        
        self._include_report = QCheckBox("Include text reports")
        self._include_report.setChecked(True)
        self._include_report.setToolTip("Generate human-readable registration reports")
        outputs_layout.addWidget(self._include_report)
        
        layout.addWidget(outputs_group)
        
        # Naming options
        naming_group = QGroupBox("File Naming")
        naming_layout = QVBoxLayout(naming_group)
        
        self._preserve_name = QCheckBox("Preserve original filename")
        self._preserve_name.setChecked(True)
        self._preserve_name.setToolTip("Use original filename with suffix added")
        naming_layout.addWidget(self._preserve_name)
        
        suffix_layout = QHBoxLayout()
        suffix_layout.addWidget(QLabel("Suffix:"))
        self._suffix_edit = QLineEdit("_registered")
        self._suffix_edit.setToolTip("Text to append to filename (e.g., 'image_registered.png')")
        suffix_layout.addWidget(self._suffix_edit)
        naming_layout.addLayout(suffix_layout)
        
        layout.addWidget(naming_group)
        
        # Output directory
        dir_group = QGroupBox("Output Directory")
        dir_layout = QHBoxLayout(dir_group)
        
        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("Select output directory...")
        dir_layout.addWidget(self._output_dir)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output_dir)
        dir_layout.addWidget(browse_btn)
        
        layout.addWidget(dir_group)
        
        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)
        
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self._cancel_btn)
        
        self._export_btn = QPushButton("Export")
        self._export_btn.clicked.connect(self._start_export)
        self._export_btn.setDefault(True)
        button_layout.addWidget(self._export_btn)
        
        layout.addLayout(button_layout)
    
    def _connect_signals(self) -> None:
        """Connect UI signals."""
        self._format_group.idToggled.connect(self._on_format_changed)
    
    def _on_format_changed(self, id: int, checked: bool) -> None:
        """Handle format selection change."""
        if checked:
            self._jpeg_quality.setEnabled(id == 2)  # JPEG
    
    def _browse_output_dir(self) -> None:
        """Open directory browser."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self._output_dir.text() or ""
        )
        if dir_path:
            self._output_dir.setText(dir_path)
    
    def _get_settings(self) -> ExportSettings:
        """Get current export settings."""
        format_map = {
            0: ExportFormat.PNG,
            1: ExportFormat.TIFF,
            2: ExportFormat.JPEG,
            3: ExportFormat.DICOM
        }
        
        return ExportSettings(
            format=format_map[self._format_group.checkedId()],
            jpeg_quality=self._jpeg_quality.value(),
            include_baseline=self._include_baseline.isChecked(),
            include_overlay=self._include_overlay.isChecked(),
            include_difference=self._include_difference.isChecked(),
            include_transform=self._include_transform.isChecked(),
            include_report=self._include_report.isChecked(),
            preserve_original_name=self._preserve_name.isChecked(),
            output_suffix=self._suffix_edit.text()
        )
    
    def _get_selected_images(self) -> List[RegisteredImage]:
        """Get list of selected images to export."""
        selected = []
        for i in range(self._image_list.count()):
            item = self._image_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(self.registered_images[i])
        return selected
    
    def _start_export(self) -> None:
        """Start the export process."""
        output_dir = self._output_dir.text()
        if not output_dir:
            QMessageBox.warning(
                self, "Warning",
                "Please select an output directory."
            )
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        selected_images = self._get_selected_images()
        if not selected_images:
            QMessageBox.warning(
                self, "Warning",
                "Please select at least one image to export."
            )
            return
        
        settings = self._get_settings()
        
        # Disable UI during export
        self._export_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        
        # Start worker
        self._worker = ExportWorker(
            selected_images,
            self.baseline,
            output_path,
            settings
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
    
    def _on_progress(self, value: int, message: str) -> None:
        """Handle progress update."""
        self._progress_bar.setValue(value)
        self._status_label.setText(message)
    
    def _on_finished(self, success: int, failed: int) -> None:
        """Handle export completion."""
        self._export_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        
        if failed == 0:
            QMessageBox.information(
                self, "Export Complete",
                f"Successfully exported {success} images to:\n{self._output_dir.text()}"
            )
            self.accept()
        else:
            QMessageBox.warning(
                self, "Export Complete",
                f"Exported {success} images.\n{failed} images failed to export.\n"
                f"Check the log for details."
            )
    
    def _on_cancel(self) -> None:
        """Handle cancel button."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()
        self.reject()
