"""
Main application window for Ophthalmic Image Registration.

Provides the primary user interface integrating all components
for longitudinal image registration and comparison.
"""

import logging
from typing import Optional
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QMenu, QToolBar, QStatusBar, QFileDialog,
    QMessageBox, QTabWidget, QApplication, QProgressDialog,
    QLabel, QFrame
)
from PyQt6.QtGui import QAction, QKeySequence, QIcon, QCloseEvent
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize

from ophthalmic_registration.gui.styles import get_theme
from ophthalmic_registration.gui.image_viewer import ImagePanel
from ophthalmic_registration.gui.comparison_view import ComparisonView
from ophthalmic_registration.gui.controls_panel import ControlsPanel, ResultsPanel
from ophthalmic_registration.gui.series_manager import SeriesManager
from ophthalmic_registration.core.image_data import ImageData, TransformResult
from ophthalmic_registration.io.image_io import ImageLoader
from ophthalmic_registration.preprocessing.pipeline import PreprocessingPipeline
from ophthalmic_registration.registration.registration_pipeline import RegistrationPipeline
from ophthalmic_registration.export.output import ExportManager

logger = logging.getLogger(__name__)


class RegistrationWorker(QThread):
    """
    Background worker for running registration.
    
    Runs the registration pipeline in a separate thread to
    keep the UI responsive.
    """
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object)  # result, registered_image
    error = pyqtSignal(str)
    
    def __init__(
        self,
        baseline: ImageData,
        followup: ImageData,
        reg_config,
        preproc_config,
        coarse_method: str = "orb",
        n_features: int = 5000,
        multimodal_options: dict = None
    ):
        super().__init__()
        self.baseline = baseline
        self.followup = followup
        self.reg_config = reg_config
        self.preproc_config = preproc_config
        self.coarse_method = coarse_method
        self.n_features = n_features
        self.multimodal_options = multimodal_options or {}
    
    def _apply_multimodal_preprocessing(self, image: ImageData, grayscale: bool, invert: bool,
                                         contrast: float = 1.0, brightness: int = 0,
                                         clahe: bool = False) -> ImageData:
        """Apply multimodality preprocessing to an image for registration."""
        import cv2
        import numpy as np
        
        if not grayscale and not invert and contrast == 1.0 and brightness == 0 and not clahe:
            return image
        
        # Get pixel array
        img = image.pixel_array.copy()
        
        # Convert to grayscale if needed
        if grayscale and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This is proven effective for multimodal retinal registration (IR↔FAF)
        if clahe:
            # Convert to grayscale if needed for CLAHE
            if img.ndim == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img
            
            # Ensure uint8 for CLAHE
            if img_gray.dtype != np.uint8:
                img_gray = (img_gray / img_gray.max() * 255).astype(np.uint8)
            
            # Apply CLAHE with parameters optimized for retinal images
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe_obj.apply(img_gray)
        
        # Apply contrast and brightness
        if contrast != 1.0 or brightness != 0:
            img = img.astype(np.float32)
            img = img * contrast + brightness
            if image.pixel_array.dtype == np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            elif image.pixel_array.dtype == np.uint16:
                img = np.clip(img, 0, 65535).astype(np.uint16)
            else:
                img = img.astype(image.pixel_array.dtype)
        
        # Invert if needed
        if invert:
            if img.dtype == np.uint8:
                img = 255 - img
            elif img.dtype == np.uint16:
                img = 65535 - img
            else:
                img = img.max() - img
        
        # Create new ImageData with modified pixel array
        modified = image.copy()
        modified._pixel_array = img
        return modified
    
    def run(self):
        try:
            from ophthalmic_registration.registration.registration_pipeline import (
                RegistrationPipeline, CoarseAlignmentMethod
            )
            
            self.progress.emit(10, "Initializing...")
            
            # Apply multimodality preprocessing for registration
            baseline_for_reg = self._apply_multimodal_preprocessing(
                self.baseline,
                self.multimodal_options.get("baseline_grayscale", False),
                self.multimodal_options.get("baseline_invert", False),
                self.multimodal_options.get("baseline_contrast", 1.0),
                self.multimodal_options.get("baseline_brightness", 0),
                self.multimodal_options.get("baseline_clahe", False)
            )
            followup_for_reg = self._apply_multimodal_preprocessing(
                self.followup,
                self.multimodal_options.get("followup_grayscale", False),
                self.multimodal_options.get("followup_invert", False),
                self.multimodal_options.get("followup_contrast", 1.0),
                self.multimodal_options.get("followup_brightness", 0),
                self.multimodal_options.get("followup_clahe", False)
            )
            
            # Create preprocessor
            preprocessor = None
            if self.preproc_config:
                preprocessor = PreprocessingPipeline(self.preproc_config)
            
            self.progress.emit(20, "Preprocessing images...")
            
            # Map string to enum
            method_map = {
                "akaze": CoarseAlignmentMethod.AKAZE,
                "orb": CoarseAlignmentMethod.ORB,
                "sift": CoarseAlignmentMethod.SIFT,
            }
            coarse_method = method_map.get(self.coarse_method, CoarseAlignmentMethod.AKAZE)
            
            # Create pipeline with selected coarse method
            pipeline = RegistrationPipeline(
                config=self.reg_config,
                preprocessor=preprocessor,
                coarse_method=coarse_method,
                n_features=self.n_features
            )
            
            self.progress.emit(40, f"Running {self.coarse_method.upper()} coarse alignment...")
            
            # Register using preprocessed images
            result = pipeline.register(
                baseline_for_reg,
                followup_for_reg,
                preprocess=self.preproc_config is not None
            )
            
            self.progress.emit(80, "Applying transform to original image...")
            
            # Apply transform to ORIGINAL follow-up image (not the preprocessed one)
            registered = pipeline.apply_transform(
                self.followup,
                result.transform_matrix,
                output_size=(self.baseline.pixel_array.shape[1], self.baseline.pixel_array.shape[0])
            )
            
            self.progress.emit(90, "Finalizing...")
            
            self.finished.emit(result, registered)
            
        except Exception as e:
            logger.exception("Registration failed")
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """
    Main application window for ophthalmic image registration.
    
    Integrates all components including:
    - Image loading and series management
    - Registration controls and configuration
    - Comparison visualization
    - Export functionality
    """
    
    def __init__(self):
        super().__init__()
        
        self._baseline: Optional[ImageData] = None
        self._followup: Optional[ImageData] = None
        self._registered: Optional[ImageData] = None
        self._result: Optional[TransformResult] = None
        self._worker: Optional[RegistrationWorker] = None
        
        # Reference baseline for longitudinal studies
        self._reference_baseline: Optional[ImageData] = None
        self._reference_locked: bool = False
        
        # Registered images history for batch export
        self._registered_images: list = []  # List of RegisteredImage objects
        
        # Undo history
        self._undo_stack: list = []
        self._max_undo = 5
        
        self._loader = ImageLoader()
        
        self._setup_window()
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()
        
        self._update_ui_state()
    
    def _setup_window(self) -> None:
        """Configure the main window."""
        self.setWindowTitle("Ophthalmic Image Registration")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Apply theme
        self.setStyleSheet(get_theme(dark=True))
    
    def _setup_ui(self) -> None:
        """Set up the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        
        # Series manager
        self._series_manager = SeriesManager()
        left_layout.addWidget(self._series_manager)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #313244; max-height: 1px;")
        left_layout.addWidget(sep)
        
        # Controls
        self._controls = ControlsPanel()
        left_layout.addWidget(self._controls, 1)
        
        main_layout.addWidget(left_panel)
        
        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("background-color: #313244; max-width: 1px;")
        main_layout.addWidget(sep2)
        
        # Center - Main view area
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        
        # Tabs for different views
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        
        # Load tab - Two image panels with overlay for registration points
        load_tab = QWidget()
        load_layout = QHBoxLayout(load_tab)
        load_layout.setContentsMargins(8, 8, 8, 8)
        load_layout.setSpacing(8)
        
        self._baseline_panel = ImagePanel("Baseline Image")
        self._baseline_panel.imageDropped.connect(self._on_baseline_dropped)
        load_layout.addWidget(self._baseline_panel, 1)
        
        self._followup_panel = ImagePanel("Follow-up Image")
        self._followup_panel.imageDropped.connect(self._on_followup_dropped)
        load_layout.addWidget(self._followup_panel, 1)
        
        self._load_tab = load_tab  # Store reference for overlay
        self._registration_lines_overlay = None  # Will be created when needed
        
        self._tabs.addTab(load_tab, "📂 Load Images")
        
        # Compare tab
        self._comparison_view = ComparisonView()
        self._tabs.addTab(self._comparison_view, "🔍 Compare")
        
        # Disable compare tab until registration is done
        self._tabs.setTabEnabled(1, False)
        
        center_layout.addWidget(self._tabs, 1)
        
        main_layout.addWidget(center_widget, 1)
        
        # Separator
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.VLine)
        sep3.setStyleSheet("background-color: #313244; max-width: 1px;")
        main_layout.addWidget(sep3)
        
        # Right panel - Results
        self._results = ResultsPanel()
        
        # Connect comparison controls to comparison view
        self._results.comparisonModeChanged.connect(self._comparison_view.set_mode)
        self._results.overlayAlphaChanged.connect(self._comparison_view.set_overlay_alpha)
        self._results.checkerSizeChanged.connect(self._comparison_view.set_checker_size)
        self._results.nativeColorChanged.connect(self._comparison_view.set_native_colors)
        self._results.fitRequested.connect(self._on_fit_requested)
        self._results.zoom100Requested.connect(self._on_zoom_100_requested)
        
        main_layout.addWidget(self._results)
    
    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_baseline = QAction("Open Baseline...", self)
        open_baseline.setShortcut(QKeySequence("Ctrl+O"))
        open_baseline.triggered.connect(self._open_baseline)
        file_menu.addAction(open_baseline)
        
        open_followup = QAction("Open Follow-up...", self)
        open_followup.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_followup.triggered.connect(self._open_followup)
        file_menu.addAction(open_followup)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export Results...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Alt+F4"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        self._undo_action = QAction("Undo Registration", self)
        self._undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        self._undo_action.triggered.connect(self._undo_registration)
        self._undo_action.setEnabled(False)
        edit_menu.addAction(self._undo_action)
        
        edit_menu.addSeparator()
        
        clear_action = QAction("Clear All", self)
        clear_action.triggered.connect(self._clear_all)
        edit_menu.addAction(clear_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        zoom_fit = QAction("Fit to View", self)
        zoom_fit.setShortcut(QKeySequence("Ctrl+0"))
        zoom_fit.triggered.connect(self._zoom_fit)
        view_menu.addAction(zoom_fit)
        
        zoom_100 = QAction("Zoom 100%", self)
        zoom_100.setShortcut(QKeySequence("Ctrl+1"))
        zoom_100.triggered.connect(self._zoom_100)
        view_menu.addAction(zoom_100)
        
        view_menu.addSeparator()
        
        self._dark_theme_action = QAction("Dark Theme", self)
        self._dark_theme_action.setCheckable(True)
        self._dark_theme_action.setChecked(True)
        self._dark_theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(self._dark_theme_action)
        
        # Registration menu
        reg_menu = menubar.addMenu("&Registration")
        
        register_action = QAction("Register Images", self)
        register_action.setShortcut(QKeySequence("Ctrl+R"))
        register_action.triggered.connect(self._run_registration)
        reg_menu.addAction(register_action)
        
        manual_action = QAction("Manual 3-Point Registration...", self)
        manual_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        manual_action.triggered.connect(self._open_manual_registration)
        reg_menu.addAction(manual_action)
        
        reg_menu.addSeparator()
        
        batch_action = QAction("Batch Registration...", self)
        batch_action.setShortcut(QKeySequence("Ctrl+B"))
        batch_action.triggered.connect(self._open_batch_registration)
        reg_menu.addAction(batch_action)
        
        reg_menu.addSeparator()
        
        self._set_reference_action = QAction("Set as Reference", self)
        self._set_reference_action.setToolTip("Lock current baseline as reference for longitudinal studies")
        self._set_reference_action.triggered.connect(self._set_as_reference)
        self._set_reference_action.setEnabled(False)
        reg_menu.addAction(self._set_reference_action)
        
        self._clear_reference_action = QAction("Clear Reference", self)
        self._clear_reference_action.setToolTip("Unlock the reference baseline")
        self._clear_reference_action.triggered.connect(self._clear_reference)
        self._clear_reference_action.setEnabled(False)
        reg_menu.addAction(self._clear_reference_action)
        
        reg_menu.addSeparator()
        
        self._batch_export_action = QAction("Batch Export Registered Images...", self)
        self._batch_export_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self._batch_export_action.setToolTip("Export all registered follow-up images")
        self._batch_export_action.triggered.connect(self._open_batch_export)
        self._batch_export_action.setEnabled(False)
        reg_menu.addAction(self._batch_export_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        measure_action = QAction("Measurement Tools", self)
        measure_action.setShortcut(QKeySequence("Ctrl+M"))
        measure_action.triggered.connect(self._toggle_measurement_tools)
        measure_action.setCheckable(True)
        tools_menu.addAction(measure_action)
        self._measure_action = measure_action
        
        tools_menu.addSeparator()
        
        self._units_action = QAction("Use mm Units (DICOM)", self)
        self._units_action.setCheckable(True)
        self._units_action.setChecked(True)  # Default to mm
        self._units_action.setToolTip("Toggle between mm and pixel units for measurements on DICOM images")
        self._units_action.triggered.connect(self._toggle_measurement_units)
        tools_menu.addAction(self._units_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        how_to_use_action = QAction("How to Use", self)
        how_to_use_action.setShortcut(QKeySequence("F1"))
        how_to_use_action.triggered.connect(self._show_how_to_use)
        help_menu.addAction(how_to_use_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_toolbar(self) -> None:
        """Set up the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Open actions
        open_baseline_btn = QAction("📂 Baseline", self)
        open_baseline_btn.setToolTip("Open baseline image")
        open_baseline_btn.triggered.connect(self._open_baseline)
        toolbar.addAction(open_baseline_btn)
        
        open_followup_btn = QAction("📁 Follow-up", self)
        open_followup_btn.setToolTip("Open follow-up image")
        open_followup_btn.triggered.connect(self._open_followup)
        toolbar.addAction(open_followup_btn)
        
        toolbar.addSeparator()
        
        # Register action
        self._register_action = QAction("▶ Register", self)
        self._register_action.setToolTip("Run registration")
        self._register_action.triggered.connect(self._run_registration)
        self._register_action.setEnabled(False)
        toolbar.addAction(self._register_action)
        
        toolbar.addSeparator()
        
        # Export action
        self._export_action = QAction("💾 Export", self)
        self._export_action.setToolTip("Export results")
        self._export_action.triggered.connect(self._export_results)
        self._export_action.setEnabled(False)
        toolbar.addAction(self._export_action)
        
        toolbar.addSeparator()
        
        # View controls
        fit_btn = QAction("⊡ Fit", self)
        fit_btn.setToolTip("Fit to view")
        fit_btn.triggered.connect(self._zoom_fit)
        toolbar.addAction(fit_btn)
        
        zoom_100_btn = QAction("1:1", self)
        zoom_100_btn.setToolTip("Zoom 100%")
        zoom_100_btn.triggered.connect(self._zoom_100)
        toolbar.addAction(zoom_100_btn)
    
    def _setup_statusbar(self) -> None:
        """Set up the status bar."""
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        
        self._status_label = QLabel("Ready")
        self._statusbar.addWidget(self._status_label, 1)
        
        self._memory_label = QLabel("")
        self._statusbar.addPermanentWidget(self._memory_label)
    
    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._controls.registerClicked.connect(self._run_registration)
        self._controls.showRegistrationPointsChanged.connect(self._on_show_registration_points)
        self._controls.showVesselContoursChanged.connect(self._on_show_vessel_contours)
        self._controls.multimodalOptionsChanged.connect(self._on_multimodal_preview)
        self._results.exportClicked.connect(self._export_results)
        
        self._series_manager.baselineChanged.connect(self._on_baseline_changed)
        self._series_manager.selectionChanged.connect(self._on_series_selection_changed)
    
    def _update_ui_state(self) -> None:
        """Update UI element states based on current data."""
        has_baseline = self._baseline is not None
        has_followup = self._followup is not None
        has_both = has_baseline and has_followup
        has_result = self._result is not None
        
        self._register_action.setEnabled(has_both)
        self._controls.set_enabled(has_both)
        self._export_action.setEnabled(has_result)
        self._tabs.setTabEnabled(1, has_result)
        
        # Reference actions
        self._set_reference_action.setEnabled(has_baseline and not self._reference_locked)
        self._clear_reference_action.setEnabled(self._reference_locked)
        
        # Batch export action
        self._batch_export_action.setEnabled(len(self._registered_images) > 0)
    
    # Supported file filters with DICOM
    FILE_FILTER = (
        "All Supported Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.dcm *.dicom);;"
        "DICOM Files (*.dcm *.dicom);;"
        "PNG Files (*.png);;"
        "JPEG Files (*.jpg *.jpeg);;"
        "TIFF Files (*.tiff *.tif);;"
        "BMP Files (*.bmp);;"
        "All Files (*)"
    )
    
    def _open_baseline(self) -> None:
        """Open file dialog for baseline image."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Baseline Image",
            "",
            self.FILE_FILTER
        )
        
        if filepath:
            self._load_baseline(filepath)
    
    def _open_followup(self) -> None:
        """Open file dialog for follow-up image."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Follow-up Image",
            "",
            self.FILE_FILTER
        )
        
        if filepath:
            self._load_followup(filepath)
    
    def _load_baseline(self, filepath: str) -> None:
        """Load baseline image from file."""
        try:
            self._baseline = self._loader.load(filepath)
            self._baseline_panel.set_image(self._baseline)
            # Fit to view by default
            self._baseline_panel.fit_to_view()
            
            # Add to series
            self._series_manager.add_image(self._baseline, filepath, "Baseline")
            
            self._status_label.setText(f"Loaded baseline: {Path(filepath).name}")
            self._update_ui_state()
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load baseline image:\n{e}"
            )
    
    def _load_followup(self, filepath: str) -> None:
        """Load follow-up image from file."""
        try:
            self._followup = self._loader.load(filepath)
            self._followup_panel.set_image(self._followup)
            # Fit to view by default
            self._followup_panel.fit_to_view()
            
            # Add to series
            self._series_manager.add_image(self._followup, filepath, "Follow-up")
            
            self._status_label.setText(f"Loaded follow-up: {Path(filepath).name}")
            self._update_ui_state()
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load follow-up image:\n{e}"
            )
    
    def _on_baseline_dropped(self, filepath: str) -> None:
        """Handle baseline image dropped."""
        self._load_baseline(filepath)
    
    def _on_followup_dropped(self, filepath: str) -> None:
        """Handle follow-up image dropped."""
        self._load_followup(filepath)
    
    def _on_baseline_changed(self, item) -> None:
        """Handle baseline selection change in series manager."""
        if item:
            self._baseline = item.image_data
            self._baseline_panel.set_image(self._baseline)
            self._update_ui_state()
    
    def _on_series_selection_changed(self, item) -> None:
        """Handle series selection change."""
        if item and not item.is_baseline:
            self._followup = item.image_data
            self._followup_panel.set_image(self._followup)
            self._update_ui_state()
    
    def _run_registration(self) -> None:
        """Run the registration pipeline."""
        if self._baseline is None or self._followup is None:
            QMessageBox.warning(
                self, "Warning",
                "Please load both baseline and follow-up images."
            )
            return
        
        # Get configuration
        reg_config = self._controls.get_registration_config()
        preproc_config = self._controls.get_preprocessing_config()
        coarse_method = self._controls.get_coarse_method()
        n_features = self._controls.get_n_features()
        multimodal_options = self._controls.get_multimodality_options()
        
        # Create and start worker
        self._worker = RegistrationWorker(
            self._baseline,
            self._followup,
            reg_config,
            preproc_config,
            coarse_method=coarse_method,
            n_features=n_features,
            multimodal_options=multimodal_options
        )
        
        self._worker.progress.connect(self._on_registration_progress)
        self._worker.finished.connect(self._on_registration_finished)
        self._worker.error.connect(self._on_registration_error)
        
        # Update UI
        self._controls.set_progress(0, True)
        self._controls.set_status("Starting registration...")
        self._controls.set_enabled(False)
        self._register_action.setEnabled(False)
        
        self._worker.start()
    
    def _on_registration_progress(self, value: int, message: str) -> None:
        """Handle registration progress update."""
        self._controls.set_progress(value)
        self._controls.set_status(message)
        self._status_label.setText(message)
    
    def _on_registration_finished(self, result: TransformResult, registered: ImageData) -> None:
        """Handle registration completion."""
        # Save to undo stack before updating
        if self._registered is not None:
            self._undo_stack.append({
                'registered': self._registered,
                'result': self._result
            })
            # Limit undo stack size
            if len(self._undo_stack) > self._max_undo:
                self._undo_stack.pop(0)
        
        self._result = result
        self._registered = registered
        
        # Add to registered images list for batch export
        from ophthalmic_registration.gui.batch_export_dialog import RegisteredImage
        visit_name = Path(self._followup.filepath).stem if self._followup.filepath else f"visit_{len(self._registered_images)+1}"
        self._registered_images.append(RegisteredImage(
            original_path=self._followup.filepath or "",
            original_image=self._followup,
            registered_image=registered,
            transform_result=result,
            visit_name=visit_name
        ))
        
        # Update UI
        self._controls.set_progress(100)
        self._controls.set_status("Registration complete!")
        self._controls.set_enabled(True)
        
        # Enable undo
        self._undo_action.setEnabled(len(self._undo_stack) > 0)
        
        # Clear and show results
        self._results.clear()
        self._results.set_result(result)
        
        # Update comparison view and switch to Overlay mode
        self._comparison_view.set_images(self._baseline, registered)
        self._comparison_view.set_registration_result(result)
        self._comparison_view.set_mode("Overlay")
        self._tabs.setTabEnabled(1, True)
        self._tabs.setCurrentIndex(1)
        
        # Update mode dropdown and fit to view
        self._results.set_comparison_mode("Overlay")
        self._comparison_view.fit_to_view()
        
        # Show method-specific status
        method = result.quality_metrics.get("alignment_method", "unknown")
        self._status_label.setText(f"Registration complete ({method.upper()})")
        
        self._update_ui_state()
        
        # Hide progress after delay
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self._controls.set_progress(0, False))
    
    def _on_registration_error(self, error_message: str) -> None:
        """Handle registration error."""
        self._controls.set_progress(0, False)
        self._controls.set_status("Registration failed")
        self._controls.set_enabled(True)
        
        self._status_label.setText("Registration failed")
        
        QMessageBox.critical(
            self, "Registration Error",
            f"Registration failed:\n{error_message}"
        )
        
        self._update_ui_state()
    
    def _undo_registration(self) -> None:
        """Undo the last registration."""
        if not self._undo_stack:
            return
        
        # Pop the last state
        prev_state = self._undo_stack.pop()
        
        self._registered = prev_state['registered']
        self._result = prev_state['result']
        
        # Update UI
        if self._registered is not None and self._result is not None:
            self._results.set_result(self._result)
            self._comparison_view.set_images(self._baseline, self._registered)
            method = self._result.quality_metrics.get("alignment_method", "unknown")
            self._status_label.setText(f"Undone to previous registration ({method.upper()})")
        else:
            self._results.clear()
            self._comparison_view.clear()
            self._tabs.setTabEnabled(1, False)
            self._tabs.setCurrentIndex(0)
            self._status_label.setText("Registration undone")
        
        # Update undo action state
        self._undo_action.setEnabled(len(self._undo_stack) > 0)
        self._update_ui_state()
    
    def _export_results(self) -> None:
        """Export registration results."""
        if self._result is None or self._registered is None:
            QMessageBox.warning(
                self, "Warning",
                "No registration results to export."
            )
            return
        
        # Get output directory
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            ""
        )
        
        if not output_dir:
            return
        
        try:
            exporter = ExportManager(output_dir=output_dir)
            
            exports = exporter.export_registration_results(
                baseline=self._baseline,
                followup=self._followup,
                registered=self._registered,
                result=self._result,
                prefix="registration",
                save_originals=True,
                save_overlay=True,
                save_difference=True
            )
            
            # Generate text report
            exporter.generate_text_report(
                self._result,
                "registration_report.txt",
                Path(output_dir)
            )
            
            self._status_label.setText(f"Exported {len(exports)} files to {output_dir}")
            
            QMessageBox.information(
                self, "Export Complete",
                f"Successfully exported {len(exports)} files to:\n{output_dir}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export results:\n{e}"
            )
    
    def _clear_all(self) -> None:
        """Clear all loaded data."""
        self._baseline = None
        self._followup = None
        self._registered = None
        self._result = None
        self._reference_baseline = None
        self._reference_locked = False
        self._registered_images = []
        
        self._baseline_panel.clear()
        self._followup_panel.clear()
        self._comparison_view.clear()
        self._results.clear()
        self._series_manager.clear()
        
        self._status_label.setText("Ready")
        self._update_ui_state()
    
    def _set_as_reference(self) -> None:
        """Set current baseline as reference for longitudinal studies."""
        if self._baseline is None:
            QMessageBox.warning(
                self, "Warning",
                "Please load a baseline image first."
            )
            return
        
        self._reference_baseline = self._baseline
        self._reference_locked = True
        
        self._baseline_panel.set_title("Baseline (Reference - Locked)")
        self._status_label.setText("Reference baseline set - load follow-up images to register")
        
        QMessageBox.information(
            self, "Reference Set",
            "The current baseline is now locked as the reference image.\n\n"
            "You can now load multiple follow-up images and register them\n"
            "against this reference. Use 'Clear Reference' to unlock."
        )
        
        self._update_ui_state()
    
    def _clear_reference(self) -> None:
        """Clear the reference baseline lock."""
        self._reference_baseline = None
        self._reference_locked = False
        
        self._baseline_panel.set_title("Baseline")
        self._status_label.setText("Reference cleared")
        
        self._update_ui_state()
    
    def _open_batch_export(self) -> None:
        """Open batch export dialog for registered images."""
        if not self._registered_images:
            QMessageBox.warning(
                self, "Warning",
                "No registered images to export.\n"
                "Please register at least one follow-up image first."
            )
            return
        
        if self._baseline is None:
            QMessageBox.warning(
                self, "Warning",
                "No baseline image loaded."
            )
            return
        
        from ophthalmic_registration.gui.batch_export_dialog import BatchExportDialog
        
        dialog = BatchExportDialog(
            self._registered_images,
            self._baseline,
            self
        )
        dialog.exec()
    
    def _zoom_fit(self) -> None:
        """Fit images to view."""
        self._baseline_panel.viewer.fit_in_view()
        self._followup_panel.viewer.fit_in_view()
    
    def _zoom_100(self) -> None:
        """Set zoom to 100%."""
        self._baseline_panel.viewer.zoom_100()
        self._followup_panel.viewer.zoom_100()
    
    def _toggle_theme(self) -> None:
        """Toggle between dark and light theme."""
        is_dark = self._dark_theme_action.isChecked()
        self.setStyleSheet(get_theme(dark=is_dark))
    
    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Ophthalmic Image Registration",
            """<h2>Ophthalmic Image Registration</h2>
            <p>Version 2.3.0</p>
            <p>A professional application for longitudinal ophthalmic image 
            registration supporting multiple imaging modalities.</p>
            <p><b>Features:</b></p>
            <ul>
                <li>DICOM and standard image format support</li>
                <li>Multiple alignment methods: AKAZE, ORB, SIFT</li>
                <li>Method-specific configuration options with tooltips</li>
                <li>Multimodality support: grayscale, invert, contrast, brightness</li>
                <li>CLAHE preprocessing for cross-modality registration (IR↔FAF)</li>
                <li>Manual 3-point registration for difficult cases</li>
                <li>Spatially distributed registration points visualization</li>
                <li>Batch processing and batch export</li>
                <li>Reference baseline for longitudinal studies</li>
                <li>Comparison modes: Side-by-Side, Overlay, Difference, Checkerboard, Split</li>
                <li>Interactive distance and area measurement tools (mm/pixels toggle)</li>
                <li>DICOM pixel spacing for accurate mm measurements</li>
                <li>Export to DICOM, PNG, TIFF, JPEG formats</li>
            </ul>
            <p>For research and clinical evaluation purposes.</p>
            <hr>
            <p><i>Created by Li Fan, 2025.12.22</i></p>
            """
        )
    
    def _show_how_to_use(self) -> None:
        """Show how-to-use help dialog with scrollable content."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton
        
        help_text = """
        <h2>How to Use Ophthalmic Image Registration</h2>
        
        <h3>Quick Start</h3>
        <ol>
            <li><b>Load Baseline:</b> File → Open Baseline (Ctrl+O) or drag & drop</li>
            <li><b>Load Follow-up:</b> File → Open Follow-up (Ctrl+Shift+O) or drag & drop</li>
            <li><b>Configure:</b> Adjust settings in the left panel if needed</li>
            <li><b>Register:</b> Click "Register Images" button or press Ctrl+R</li>
            <li><b>Compare:</b> Switch to "Compare" tab to view results</li>
        </ol>
        
        <h3>Alignment Methods</h3>
        <p>Each method has its own configuration options:</p>
        <ul>
            <li><b>AKAZE (Recommended):</b> Best for most ophthalmic images
                <ul>
                    <li>Threshold: Controls feature detection sensitivity</li>
                    <li>Match Ratio: Strictness of feature matching</li>
                </ul>
            </li>
            <li><b>ORB:</b> Fastest option, good for quick previews
                <ul>
                    <li>Max Features: Number of features to detect</li>
                    <li>Match Ratio: Strictness of feature matching</li>
                </ul>
            </li>
            <li><b>SIFT:</b> Classic algorithm, very accurate for detailed images
                <ul>
                    <li>Max Features: Number of features to detect</li>
                    <li>Match Ratio: Strictness of feature matching</li>
                </ul>
            </li>
        </ul>
        
        <h3>Multimodality Registration (IR↔FAF)</h3>
        <p>When registering images from different modalities (e.g., FAF vs IR), 
        use the <b>Multimodality Options</b> to improve alignment:</p>
        <ul>
            <li><b>Grayscale:</b> Convert color fundus images to grayscale</li>
            <li><b>Invert:</b> Invert image colors to match vessel polarity
                <ul>
                    <li>FAF images: vessels appear <b>dark</b></li>
                    <li>IR images: vessels appear <b>bright</b></li>
                    <li>Invert one image so vessels match in both</li>
                </ul>
            </li>
            <li><b>Contrast/Brightness:</b> Adjust image intensity</li>
            <li><b>CLAHE:</b> Contrast Limited Adaptive Histogram Equalization
                <ul>
                    <li>Recommended for cross-modality registration</li>
                    <li>Enhances local contrast for better feature matching</li>
                </ul>
            </li>
        </ul>
        <p><i>Note: These options affect registration only. Results display in original colors.</i></p>
        
        <h3>Registration Points Visualization</h3>
        <p>To see the feature points used for alignment:</p>
        <ol>
            <li>Run registration first</li>
            <li>Check "Show Registration Points" in the Visualization section</li>
            <li>View up to 50 spatially distributed points across the image</li>
            <li>Points are color-coded with connecting lines showing correspondence</li>
        </ol>
        
        <h3>Reference Baseline (Longitudinal Studies)</h3>
        <p>For studies with multiple visits:</p>
        <ol>
            <li>Load your reference baseline (usually first visit)</li>
            <li>Go to <b>Registration → Set as Reference</b></li>
            <li>The baseline is locked for all subsequent registrations</li>
            <li>Load and register follow-up images from different visits</li>
            <li>Use <b>Registration → Clear Reference</b> to unlock</li>
        </ol>
        
        <h3>Batch Processing</h3>
        <ol>
            <li>Load your baseline image first</li>
            <li>Go to <b>Registration → Batch Registration</b></li>
            <li>Select multiple follow-up images</li>
            <li>Choose output directory</li>
            <li>Click "Start Batch" to process all</li>
        </ol>
        
        <h3>Manual 3-Point Registration</h3>
        <p>For difficult cases where automatic methods fail:</p>
        <ol>
            <li>Go to <b>Registration → Manual 3-Point Registration</b></li>
            <li>Click on 3 corresponding landmarks in both images</li>
            <li>The transform is computed from these points</li>
        </ol>
        
        <h3>Measurement Tools</h3>
        <ol>
            <li><b>Right-click</b> on any image viewer</li>
            <li>Select <b>Measure → Distance</b> or <b>Measure → Area</b></li>
            <li>For distance: click two points</li>
            <li>For area: drag to draw polygon, release to complete</li>
            <li>Right-click on measurement to delete</li>
            <li>Measurements show in mm if DICOM pixel spacing is available</li>
        </ol>
        
        <h3>Comparison Modes</h3>
        <ul>
            <li><b>Side by Side:</b> View baseline and registered images together</li>
            <li><b>Overlay:</b> Blend images with adjustable transparency</li>
            <li><b>Difference:</b> Highlight changes between images</li>
            <li><b>Checkerboard:</b> Alternating tiles for alignment verification</li>
            <li><b>Split View:</b> Draggable divider between images</li>
        </ul>
        
        <h3>Keyboard Shortcuts</h3>
        <ul>
            <li><b>Ctrl+O:</b> Open baseline image</li>
            <li><b>Ctrl+Shift+O:</b> Open follow-up image</li>
            <li><b>Ctrl+R:</b> Run registration</li>
            <li><b>Ctrl+Z:</b> Undo last registration</li>
            <li><b>Ctrl+E:</b> Export results</li>
            <li><b>Ctrl+Shift+E:</b> Batch export</li>
        </ul>
        
        <h3>Tips</h3>
        <ul>
            <li>Hover over any setting to see a tooltip explanation</li>
            <li>Use CLAHE preprocessing for cross-modality registration (IR↔FAF)</li>
            <li>For cross-modality, also try Invert option to match vessel polarity</li>
            <li>Check registration quality score - "Failed" means alignment didn't work</li>
            <li>DICOM images preserve pixel spacing for accurate measurements</li>
            <li>Toggle measurement units (mm/pixels) via Tools menu</li>
        </ul>
        """
        
        # Create scrollable dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("How to Use")
        dialog.setMinimumSize(600, 500)
        dialog.resize(700, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Scrollable text browser
        text_browser = QTextBrowser()
        text_browser.setHtml(help_text)
        text_browser.setOpenExternalLinks(True)
        layout.addWidget(text_browser)
        
        # OK button
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        layout.addWidget(ok_btn)
        
        dialog.exec()
    
    def _open_manual_registration(self) -> None:
        """Open manual 3-point registration dialog."""
        if self._baseline is None or self._followup is None:
            QMessageBox.warning(
                self, "Warning",
                "Please load both baseline and follow-up images first."
            )
            return
        
        from ophthalmic_registration.gui.manual_registration import ManualRegistrationDialog
        
        dialog = ManualRegistrationDialog(
            self._baseline,
            self._followup,
            self
        )
        
        dialog.registrationComplete.connect(self._on_manual_registration_complete)
        dialog.exec()
    
    def _on_manual_registration_complete(self, result, registered) -> None:
        """Handle manual registration completion."""
        # Save to undo stack before updating
        if self._registered is not None:
            self._undo_stack.append({
                'registered': self._registered,
                'result': self._result
            })
            if len(self._undo_stack) > self._max_undo:
                self._undo_stack.pop(0)
        
        self._result = result
        self._registered = registered
        
        # Clear and update results panel
        self._results.clear()
        self._results.set_result(result)
        
        # Enable undo
        self._undo_action.setEnabled(len(self._undo_stack) > 0)
        
        # Update comparison view
        self._comparison_view.set_images(self._baseline, registered)
        self._tabs.setTabEnabled(1, True)
        self._tabs.setCurrentIndex(1)
        
        self._status_label.setText("Manual 3-point registration complete")
        self._update_ui_state()
    
    def _open_batch_registration(self) -> None:
        """Open batch registration dialog."""
        if self._baseline is None:
            QMessageBox.warning(
                self, "Warning",
                "Please load a baseline image first."
            )
            return
        
        from ophthalmic_registration.gui.batch_registration import BatchRegistrationDialog
        
        reg_config = self._controls.get_registration_config()
        preproc_config = self._controls.get_preprocessing_config()
        
        dialog = BatchRegistrationDialog(
            self._baseline,
            reg_config,
            preproc_config,
            self
        )
        
        dialog.registrationCompleted.connect(self._on_batch_completed)
        dialog.exec()
    
    def _on_batch_completed(self, results: list) -> None:
        """Handle batch registration completion."""
        self._status_label.setText(
            f"Batch registration complete: {len(results)} images processed"
        )
        
        # Update series manager with results
        for result, registered, item in results:
            # Find and update the item in series manager
            for i, series_item in enumerate(self._series_manager.get_all_items()):
                if series_item.filepath == item.filepath:
                    self._series_manager.mark_registered(i, result)
                    break
    
    def _toggle_measurement_tools(self) -> None:
        """Toggle measurement tools - now shows help message since tools are in right-click menu."""
        is_visible = self._measure_action.isChecked()
        
        if is_visible:
            self._status_label.setText("Right-click on any image to access Measure > Distance or Area tools")
        else:
            self._status_label.setText("Ready")
    
    def _toggle_measurement_units(self) -> None:
        """Toggle measurement units between mm and pixels."""
        from ophthalmic_registration.gui.measurement_items_new import DistanceMeasurement, AreaMeasurement
        
        use_mm = self._units_action.isChecked()
        DistanceMeasurement.use_mm_units = use_mm
        AreaMeasurement.use_mm_units = use_mm
        
        # Update all existing measurements
        for panel in [self._baseline_panel, self._followup_panel]:
            if hasattr(panel, 'viewer') and hasattr(panel.viewer, '_measurements'):
                for measurement in panel.viewer._measurements:
                    measurement.update_from_points()
        
        # Update comparison view measurements if any
        if hasattr(self._comparison_view, '_measurements'):
            for measurement in self._comparison_view._measurements:
                measurement.update_from_points()
        
        unit_text = "mm" if use_mm else "pixels"
        self._status_label.setText(f"Measurement units set to {unit_text}")
    
    def _on_fit_requested(self) -> None:
        """Handle Fit button click."""
        if self._tabs.currentIndex() == 1:  # Compare tab
            self._comparison_view.fit_to_view()
        else:  # Load Images tab
            self._baseline_panel.fit_to_view()
            self._followup_panel.fit_to_view()
    
    def _on_zoom_100_requested(self) -> None:
        """Handle 1:1 zoom button click."""
        if self._tabs.currentIndex() == 1:  # Compare tab
            self._comparison_view.zoom_100()
        else:  # Load Images tab
            self._baseline_panel.zoom_100()
            self._followup_panel.zoom_100()
    
    def _on_multimodal_preview(self, options: dict) -> None:
        """Update live preview of multimodality options on Load Images tab."""
        import cv2
        import numpy as np
        
        def apply_preview(img_data, grayscale, invert, contrast, brightness, clahe):
            """Apply multimodality adjustments to numpy array for preview."""
            if img_data is None:
                return None
            
            img = img_data.as_uint8()
            
            # Convert to grayscale if needed
            if grayscale and img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Back to RGB for display
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if clahe:
                if img.ndim == 3:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = img
                
                # Apply CLAHE with parameters optimized for retinal images
                clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_clahe = clahe_obj.apply(img_gray)
                img = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
            
            # Apply contrast and brightness
            if contrast != 1.0 or brightness != 0:
                img = img.astype(np.float32)
                img = img * contrast + brightness
                img = np.clip(img, 0, 255).astype(np.uint8)
            
            # Invert if needed
            if invert:
                img = 255 - img
            
            return np.ascontiguousarray(img)
        
        # Check if any options are non-default
        has_baseline_changes = (options.get("baseline_grayscale", False) or 
                               options.get("baseline_invert", False) or
                               options.get("baseline_contrast", 1.0) != 1.0 or
                               options.get("baseline_brightness", 0) != 0 or
                               options.get("baseline_clahe", False))
        
        has_followup_changes = (options.get("followup_grayscale", False) or 
                               options.get("followup_invert", False) or
                               options.get("followup_contrast", 1.0) != 1.0 or
                               options.get("followup_brightness", 0) != 0 or
                               options.get("followup_clahe", False))
        
        # Update baseline preview
        if self._baseline:
            if has_baseline_changes:
                preview = apply_preview(
                    self._baseline,
                    options.get("baseline_grayscale", False),
                    options.get("baseline_invert", False),
                    options.get("baseline_contrast", 1.0),
                    options.get("baseline_brightness", 0),
                    options.get("baseline_clahe", False)
                )
                if preview is not None:
                    self._baseline_panel.viewer.set_numpy_array(preview)
            else:
                self._baseline_panel.set_image(self._baseline)
        
        # Update follow-up preview
        if self._followup:
            if has_followup_changes:
                preview = apply_preview(
                    self._followup,
                    options.get("followup_grayscale", False),
                    options.get("followup_invert", False),
                    options.get("followup_contrast", 1.0),
                    options.get("followup_brightness", 0),
                    options.get("followup_clahe", False)
                )
                if preview is not None:
                    self._followup_panel.viewer.set_numpy_array(preview)
            else:
                self._followup_panel.set_image(self._followup)
    
    def _on_show_vessel_contours(self, show: bool) -> None:
        """Show or hide vessel contour overlays on the Load Images tab."""
        if not show:
            # Restore original images
            if self._baseline:
                self._baseline_panel.set_image(self._baseline)
            if self._followup:
                self._followup_panel.set_image(self._followup)
            return
        
        if not self._baseline and not self._followup:
            self._status_label.setText("Load images first to show vessel contours")
            return
        
        try:
            import cv2
            import numpy as np
            from skimage.filters import frangi
            from skimage.morphology import skeletonize
            
            def extract_vessel_contours(img_data, color=(0, 255, 0)):
                """Extract and overlay vessel contours on image."""
                if img_data is None:
                    return None
                
                img = img_data.as_uint8()
                
                # Convert to grayscale for vessel detection
                if img.ndim == 3:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = img
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                # Normalize to 0-1 for Frangi
                img_norm = img_gray.astype(np.float64) / 255.0
                
                # Apply Frangi filter
                vessel_response = frangi(img_norm, sigmas=range(1, 8, 2), black_ridges=False)
                
                # Threshold to get binary vessel mask
                if vessel_response.max() > 0:
                    vessel_norm = vessel_response / vessel_response.max()
                    vessel_mask = (vessel_norm > 0.1).astype(np.uint8) * 255
                else:
                    vessel_mask = np.zeros_like(img_gray, dtype=np.uint8)
                
                # Find contours
                contours, _ = cv2.findContours(vessel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours on original image
                result = img.copy()
                cv2.drawContours(result, contours, -1, color, 1)
                
                # Also draw skeletonized vessels for centerline visualization
                if vessel_mask.max() > 0:
                    skeleton = skeletonize(vessel_mask > 0)
                    # Draw skeleton in a brighter color
                    skeleton_color = (255, 255, 0)  # Yellow for centerlines
                    result[skeleton] = skeleton_color
                
                return np.ascontiguousarray(result)
            
            self._status_label.setText("Extracting vessel contours...")
            
            # Process baseline
            if self._baseline:
                baseline_with_contours = extract_vessel_contours(
                    self._baseline, 
                    color=(0, 255, 0)  # Green for baseline
                )
                if baseline_with_contours is not None:
                    self._baseline_panel.viewer.set_numpy_array(baseline_with_contours)
            
            # Process follow-up
            if self._followup:
                followup_with_contours = extract_vessel_contours(
                    self._followup,
                    color=(255, 0, 255)  # Magenta for follow-up
                )
                if followup_with_contours is not None:
                    self._followup_panel.viewer.set_numpy_array(followup_with_contours)
            
            self._status_label.setText("Vessel contours displayed (Green=baseline, Magenta=follow-up, Yellow=centerlines)")
            
        except Exception as e:
            logger.error(f"Failed to extract vessel contours: {e}")
            self._status_label.setText(f"Failed to extract vessel contours: {e}")
    
    def _on_show_registration_points(self, show: bool) -> None:
        """Show or hide registration points on the Load Images tab."""
        if not show:
            # Restore original images
            if self._baseline:
                self._baseline_panel.set_image(self._baseline)
            if self._followup:
                self._followup_panel.set_image(self._followup)
            return
        
        # Check if we have registration result with feature matches
        if not self._result or not self._result.feature_match_result:
            self._status_label.setText("No registration points available - run registration first")
            return
        
        feature_result = self._result.feature_match_result
        if feature_result.inlier_mask is None or len(feature_result.matches) == 0:
            self._status_label.setText("No feature matches available")
            return
        
        try:
            import cv2
            import numpy as np
            
            # Get images
            img1 = self._baseline.as_uint8()
            img2 = self._followup.as_uint8()
            
            # Convert to RGB if grayscale
            if img1.ndim == 2:
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
            else:
                img1 = img1.copy()
            
            if img2.ndim == 2:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
            else:
                img2 = img2.copy()
            
            # Get inlier matches with their coordinates
            inlier_matches = []
            for i, match in enumerate(feature_result.matches):
                if feature_result.inlier_mask[i]:
                    pt = feature_result.keypoints_baseline[match.queryIdx].pt
                    inlier_matches.append((i, match, match.distance, pt[0], pt[1]))
            
            # Get image dimensions
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # Select spatially distributed points across the image
            # Divide image into a 5x5 grid and randomly select points from each cell
            import random
            grid_rows, grid_cols = 5, 5
            cell_h, cell_w = h1 / grid_rows, w1 / grid_cols
            
            # Group matches by grid cell
            grid_cells = {}
            for match_data in inlier_matches:
                i, match, dist, x, y = match_data
                cell_row = min(int(y / cell_h), grid_rows - 1)
                cell_col = min(int(x / cell_w), grid_cols - 1)
                cell_key = (cell_row, cell_col)
                if cell_key not in grid_cells:
                    grid_cells[cell_key] = []
                grid_cells[cell_key].append((i, match, dist))
            
            # Randomly select one point from each cell that has matches
            top_matches = []
            for cell_key in grid_cells:
                cell_matches = grid_cells[cell_key]
                # Randomly select one match from this cell
                selected = random.choice(cell_matches)
                top_matches.append(selected)
            
            # Limit to 50 if we have more cells with matches
            if len(top_matches) > 50:
                top_matches = random.sample(top_matches, 50)
            
            # Create combined image for drawing continuous lines
            max_h = max(h1, h2)
            gap = 20  # Gap between images to represent the panel spacing
            combined = np.zeros((max_h, w1 + gap + w2, 3), dtype=np.uint8)
            combined[:h1, :w1] = img1
            combined[:, w1:w1+gap] = 40  # Dark gray gap
            combined[:h2, w1+gap:] = img2
            
            # Draw points and continuous connecting lines on combined image
            for idx, (i, match, dist) in enumerate(top_matches):
                pt1 = feature_result.keypoints_baseline[match.queryIdx].pt
                pt2 = feature_result.keypoints_followup[match.trainIdx].pt
                
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]) + w1 + gap, int(pt2[1])  # Offset for combined image
                
                # Generate a unique color for each match pair
                color = self._get_match_color(idx)
                
                # Draw continuous connecting line across both images
                cv2.line(combined, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                
                # Draw circles at keypoints
                cv2.circle(combined, (x1, y1), 6, color, -1)
                cv2.circle(combined, (x1, y1), 8, (255, 255, 255), 2)
                cv2.putText(combined, str(idx + 1), (x1 + 10, y1 + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(combined, str(idx + 1), (x1 + 10, y1 + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                cv2.circle(combined, (x2, y2), 6, color, -1)
                cv2.circle(combined, (x2, y2), 8, (255, 255, 255), 2)
                cv2.putText(combined, str(idx + 1), (x2 + 10, y2 + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(combined, str(idx + 1), (x2 + 10, y2 + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Split combined image back into two parts
            img1_annotated = np.ascontiguousarray(combined[:h1, :w1])
            img2_annotated = np.ascontiguousarray(combined[:h2, w1+gap:])
            
            # Update panels with annotated images
            self._baseline_panel.viewer.set_numpy_array(img1_annotated)
            self._followup_panel.viewer.set_numpy_array(img2_annotated)
            
            self._status_label.setText(f"Showing {len(top_matches)} spatially distributed registration points")
            
        except Exception as e:
            self._status_label.setText(f"Error showing registration points: {str(e)}")
    
    def _get_match_color(self, idx: int) -> tuple:
        """Get a distinct color for a match index."""
        colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (0, 255, 128),    # Spring green
            (255, 0, 128),    # Rose
            (128, 255, 0),    # Lime
            (0, 128, 255),    # Sky blue
            (255, 128, 128),  # Light red
            (128, 255, 128),  # Light green
            (128, 128, 255),  # Light blue
            (255, 255, 128),  # Light yellow
            (255, 128, 255),  # Light magenta
            (128, 255, 255),  # Light cyan
            (192, 64, 0),     # Brown
            (64, 192, 0),     # Dark lime
            (0, 64, 192),     # Dark blue
            (192, 0, 64),     # Dark rose
            (64, 0, 192),     # Indigo
            (0, 192, 64),     # Teal
            (192, 192, 0),    # Olive
        ]
        return colors[idx % len(colors)]
    
    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close event."""
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Registration is in progress. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            
            self._worker.terminate()
            self._worker.wait()
        
        event.accept()
