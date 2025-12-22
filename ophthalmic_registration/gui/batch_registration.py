"""
Batch registration for multiple follow-up images.

Provides functionality to register multiple follow-up images
against a single baseline image in sequence.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QProgressBar, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QDialog, QDialogButtonBox,
    QCheckBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread

from ophthalmic_registration.core.image_data import ImageData, TransformResult
from ophthalmic_registration.io.image_io import ImageLoader
from ophthalmic_registration.preprocessing.pipeline import PreprocessingPipeline
from ophthalmic_registration.registration.registration_pipeline import RegistrationPipeline


@dataclass
class BatchItem:
    """Represents an item in the batch registration queue."""
    filepath: str
    filename: str
    image_data: Optional[ImageData] = None
    result: Optional[TransformResult] = None
    registered: Optional[ImageData] = None
    status: str = "pending"  # pending, processing, completed, failed
    error_message: str = ""


class BatchRegistrationWorker(QThread):
    """
    Background worker for batch registration.
    
    Registers multiple follow-up images against a baseline.
    """
    
    progress = pyqtSignal(int, str)  # overall progress, message
    itemStarted = pyqtSignal(int)  # item index
    itemCompleted = pyqtSignal(int, object, object)  # index, result, registered
    itemFailed = pyqtSignal(int, str)  # index, error message
    finished = pyqtSignal()
    
    def __init__(
        self,
        baseline: ImageData,
        items: List[BatchItem],
        reg_config,
        preproc_config
    ):
        super().__init__()
        self.baseline = baseline
        self.items = items
        self.reg_config = reg_config
        self.preproc_config = preproc_config
        self._cancelled = False
    
    def run(self):
        """Run batch registration."""
        # Create pipeline
        preprocessor = None
        if self.preproc_config:
            preprocessor = PreprocessingPipeline(self.preproc_config)
        
        pipeline = RegistrationPipeline(
            config=self.reg_config,
            preprocessor=preprocessor
        )
        
        total = len(self.items)
        
        for i, item in enumerate(self.items):
            if self._cancelled:
                break
            
            self.itemStarted.emit(i)
            self.progress.emit(
                int((i / total) * 100),
                f"Processing {item.filename} ({i+1}/{total})"
            )
            
            try:
                # Load image if not already loaded
                if item.image_data is None:
                    loader = ImageLoader()
                    item.image_data = loader.load(item.filepath)
                
                # Register
                result, registered = pipeline.register_and_apply(
                    self.baseline,
                    item.image_data,
                    preprocess=self.preproc_config is not None
                )
                
                self.itemCompleted.emit(i, result, registered)
                
            except Exception as e:
                self.itemFailed.emit(i, str(e))
        
        self.progress.emit(100, "Batch registration complete")
        self.finished.emit()
    
    def cancel(self):
        """Cancel the batch registration."""
        self._cancelled = True


class BatchRegistrationDialog(QDialog):
    """
    Dialog for batch registration of multiple follow-up images.
    
    Allows users to select multiple images and register them
    all against a single baseline.
    """
    
    # File filter with DICOM support
    FILE_FILTER = (
        "All Supported Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.dcm *.dicom);;"
        "DICOM Files (*.dcm *.dicom);;"
        "PNG Files (*.png);;"
        "JPEG Files (*.jpg *.jpeg);;"
        "TIFF Files (*.tiff *.tif);;"
        "BMP Files (*.bmp);;"
        "All Files (*)"
    )
    
    registrationCompleted = pyqtSignal(list)  # List of (result, registered) tuples
    
    def __init__(
        self,
        baseline: ImageData,
        reg_config,
        preproc_config,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self._baseline = baseline
        self._reg_config = reg_config
        self._preproc_config = preproc_config
        self._items: List[BatchItem] = []
        self._worker: Optional[BatchRegistrationWorker] = None
        self._results: List[tuple] = []
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        self.setWindowTitle("Batch Registration")
        self.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Info
        info_label = QLabel(
            "Register multiple follow-up images against the loaded baseline.\n"
            "All images will use the same registration settings."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # File list
        files_group = QGroupBox("Follow-up Images")
        files_layout = QVBoxLayout(files_group)
        
        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        files_layout.addWidget(self._file_list)
        
        # File buttons
        file_btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add Files...")
        add_btn.clicked.connect(self._add_files)
        file_btn_layout.addWidget(add_btn)
        
        add_folder_btn = QPushButton("Add Folder...")
        add_folder_btn.clicked.connect(self._add_folder)
        file_btn_layout.addWidget(add_folder_btn)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        file_btn_layout.addWidget(remove_btn)
        
        file_btn_layout.addStretch()
        files_layout.addLayout(file_btn_layout)
        
        layout.addWidget(files_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        progress_layout.addWidget(self._progress_bar)
        
        self._status_label = QLabel("Ready")
        progress_layout.addWidget(self._status_label)
        
        layout.addWidget(progress_group)
        
        # Options
        options_layout = QHBoxLayout()
        
        self._export_check = QCheckBox("Export results after completion")
        self._export_check.setChecked(True)
        options_layout.addWidget(self._export_check)
        
        options_layout.addStretch()
        layout.addLayout(options_layout)
        
        # Buttons
        button_box = QDialogButtonBox()
        
        self._start_btn = QPushButton("Start Registration")
        self._start_btn.clicked.connect(self._start_registration)
        self._start_btn.setEnabled(False)
        button_box.addButton(self._start_btn, QDialogButtonBox.ButtonRole.ActionRole)
        
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._cancel)
        button_box.addButton(self._cancel_btn, QDialogButtonBox.ButtonRole.RejectRole)
        
        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.accept)
        self._close_btn.setVisible(False)
        button_box.addButton(self._close_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        
        layout.addWidget(button_box)
    
    def _add_files(self) -> None:
        """Add files to the batch."""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Follow-up Images",
            "",
            self.FILE_FILTER
        )
        
        for filepath in filepaths:
            self._add_item(filepath)
        
        self._update_ui()
    
    def _add_folder(self) -> None:
        """Add all images from a folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with Follow-up Images"
        )
        
        if folder:
            folder_path = Path(folder)
            extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.dcm', '.dicom'}
            
            for filepath in folder_path.iterdir():
                if filepath.suffix.lower() in extensions:
                    self._add_item(str(filepath))
        
        self._update_ui()
    
    def _add_item(self, filepath: str) -> None:
        """Add a single item to the batch."""
        # Check if already added
        for item in self._items:
            if item.filepath == filepath:
                return
        
        item = BatchItem(
            filepath=filepath,
            filename=Path(filepath).name
        )
        self._items.append(item)
        
        list_item = QListWidgetItem(item.filename)
        list_item.setData(Qt.ItemDataRole.UserRole, len(self._items) - 1)
        self._file_list.addItem(list_item)
    
    def _remove_selected(self) -> None:
        """Remove selected items from the batch."""
        selected = self._file_list.selectedItems()
        
        # Get indices to remove (in reverse order to maintain indices)
        indices = sorted(
            [item.data(Qt.ItemDataRole.UserRole) for item in selected],
            reverse=True
        )
        
        for idx in indices:
            if 0 <= idx < len(self._items):
                del self._items[idx]
        
        # Rebuild list
        self._file_list.clear()
        for i, item in enumerate(self._items):
            list_item = QListWidgetItem(item.filename)
            list_item.setData(Qt.ItemDataRole.UserRole, i)
            self._file_list.addItem(list_item)
        
        self._update_ui()
    
    def _update_ui(self) -> None:
        """Update UI state."""
        has_items = len(self._items) > 0
        self._start_btn.setEnabled(has_items)
        self._status_label.setText(f"{len(self._items)} images queued")
    
    def _start_registration(self) -> None:
        """Start the batch registration."""
        if not self._items:
            return
        
        # Reset status
        for item in self._items:
            item.status = "pending"
            item.result = None
            item.registered = None
            item.error_message = ""
        
        self._results = []
        
        # Update UI
        self._start_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        
        # Create and start worker
        self._worker = BatchRegistrationWorker(
            self._baseline,
            self._items,
            self._reg_config,
            self._preproc_config
        )
        
        self._worker.progress.connect(self._on_progress)
        self._worker.itemStarted.connect(self._on_item_started)
        self._worker.itemCompleted.connect(self._on_item_completed)
        self._worker.itemFailed.connect(self._on_item_failed)
        self._worker.finished.connect(self._on_finished)
        
        self._worker.start()
    
    def _on_progress(self, value: int, message: str) -> None:
        """Handle progress update."""
        self._progress_bar.setValue(value)
        self._status_label.setText(message)
    
    def _on_item_started(self, index: int) -> None:
        """Handle item started."""
        self._items[index].status = "processing"
        
        item = self._file_list.item(index)
        if item:
            item.setText(f"⏳ {self._items[index].filename}")
    
    def _on_item_completed(self, index: int, result, registered) -> None:
        """Handle item completed."""
        self._items[index].status = "completed"
        self._items[index].result = result
        self._items[index].registered = registered
        
        self._results.append((result, registered, self._items[index]))
        
        item = self._file_list.item(index)
        if item:
            ecc = result.ecc_correlation or 0
            item.setText(f"✓ {self._items[index].filename} (ECC: {ecc:.3f})")
            item.setForeground(Qt.GlobalColor.green)
    
    def _on_item_failed(self, index: int, error: str) -> None:
        """Handle item failed."""
        self._items[index].status = "failed"
        self._items[index].error_message = error
        
        item = self._file_list.item(index)
        if item:
            item.setText(f"✗ {self._items[index].filename}")
            item.setForeground(Qt.GlobalColor.red)
    
    def _on_finished(self) -> None:
        """Handle batch completion."""
        self._start_btn.setEnabled(True)
        self._close_btn.setVisible(True)
        
        # Count results
        completed = sum(1 for item in self._items if item.status == "completed")
        failed = sum(1 for item in self._items if item.status == "failed")
        
        self._status_label.setText(
            f"Complete: {completed} succeeded, {failed} failed"
        )
        
        # Emit results
        if self._results:
            self.registrationCompleted.emit(self._results)
        
        # Export if requested
        if self._export_check.isChecked() and self._results:
            self._export_results()
    
    def _export_results(self) -> None:
        """Export batch registration results."""
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory"
        )
        
        if not output_dir:
            return
        
        from ophthalmic_registration.export.output import ExportManager
        
        try:
            exporter = ExportManager(output_dir=output_dir)
            
            for i, (result, registered, item) in enumerate(self._results):
                prefix = Path(item.filename).stem
                
                # Save registered image
                exporter.save_image(
                    registered,
                    f"{prefix}_registered.png"
                )
                
                # Save transform
                exporter.save_transform(
                    result,
                    f"{prefix}_transform.json"
                )
            
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(self._results)} registration results to:\n{output_dir}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export results:\n{e}"
            )
    
    def _cancel(self) -> None:
        """Cancel and close dialog."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()
        
        self.reject()
    
    def get_results(self) -> List[tuple]:
        """Get the registration results."""
        return self._results
