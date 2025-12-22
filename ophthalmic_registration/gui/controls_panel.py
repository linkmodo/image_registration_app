"""
Registration controls panel.

Provides UI controls for configuring and running the registration pipeline.
"""

from typing import Optional
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QProgressBar, QFrame, QScrollArea,
    QSizePolicy, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal

from ophthalmic_registration.core.image_data import (
    RegistrationConfig,
    MotionModel,
)
from ophthalmic_registration.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingStep,
)


class ControlsPanel(QWidget):
    """
    Controls panel for registration settings.
    
    Provides UI for configuring preprocessing and registration parameters.
    
    Signals:
        registerClicked: Emitted when register button is clicked
        configChanged: Emitted when any configuration changes
    """
    
    registerClicked = pyqtSignal()
    configChanged = pyqtSignal()
    showRegistrationPointsChanged = pyqtSignal(bool)
    showVesselContoursChanged = pyqtSignal(bool)  # Emitted when vessel contour visualization changes
    multimodalOptionsChanged = pyqtSignal(dict)  # Emitted when multimodality preview options change
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.setFixedWidth(320)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the controls UI."""
        # Main scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("Registration Settings")
        title.setObjectName("titleLabel")
        layout.addWidget(title)
        
        # Preprocessing group
        preproc_group = self._create_preprocessing_group()
        layout.addWidget(preproc_group)
        
        # Multimodality preprocessing group
        multimodal_group = self._create_multimodality_group()
        layout.addWidget(multimodal_group)
        
        # Registration group
        reg_group = self._create_registration_group()
        layout.addWidget(reg_group)
        
        # Validation group
        valid_group = self._create_validation_group()
        layout.addWidget(valid_group)
        
        # Visualization options
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)
        viz_layout.setSpacing(6)
        
        # Registration points
        points_layout = QHBoxLayout()
        points_layout.addWidget(QLabel("Show Registration Points:"))
        self._show_reg_points_checkbox = QCheckBox()
        self._show_reg_points_checkbox.setChecked(False)
        self._show_reg_points_checkbox.setToolTip(
            "Show the feature points used for registration\n"
            "on the baseline and follow-up images."
        )
        self._show_reg_points_checkbox.stateChanged.connect(self._on_show_reg_points_changed)
        points_layout.addWidget(self._show_reg_points_checkbox)
        points_layout.addStretch()
        viz_layout.addLayout(points_layout)
        
        # Vessel contour overlay
        vessel_layout = QHBoxLayout()
        vessel_layout.addWidget(QLabel("Show Vessel Contours:"))
        self._show_vessel_contours_checkbox = QCheckBox()
        self._show_vessel_contours_checkbox.setChecked(False)
        self._show_vessel_contours_checkbox.setToolTip(
            "Show detected blood vessel contours as colored\n"
            "overlays on the baseline and follow-up images.\n"
            "Useful for visualizing vessel alignment."
        )
        self._show_vessel_contours_checkbox.stateChanged.connect(self._on_show_vessel_contours_changed)
        vessel_layout.addWidget(self._show_vessel_contours_checkbox)
        vessel_layout.addStretch()
        viz_layout.addLayout(vessel_layout)
        
        layout.addWidget(viz_group)
        
        # Spacer
        layout.addStretch()
        
        # Action buttons
        action_layout = QVBoxLayout()
        action_layout.setSpacing(8)
        
        self._register_btn = QPushButton("▶  Register Images")
        self._register_btn.setMinimumHeight(44)
        self._register_btn.clicked.connect(self.registerClicked.emit)
        action_layout.addWidget(self._register_btn)
        
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(False)
        action_layout.addWidget(self._progress_bar)
        
        self._status_label = QLabel("Ready")
        self._status_label.setObjectName("subtitleLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        action_layout.addWidget(self._status_label)
        
        layout.addLayout(action_layout)
        
        scroll.setWidget(container)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
    
    def _create_preprocessing_group(self) -> QGroupBox:
        """Create preprocessing settings group."""
        group = QGroupBox("Preprocessing")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        
        # Enable preprocessing
        self._preproc_enabled = QCheckBox("Enable Preprocessing")
        self._preproc_enabled.setChecked(True)
        self._preproc_enabled.setToolTip(
            "Apply image enhancement before registration.\n"
            "Improves feature detection and matching quality.\n"
            "Recommended for most images."
        )
        self._preproc_enabled.stateChanged.connect(self._on_config_changed)
        layout.addWidget(self._preproc_enabled)
        
        # CLAHE
        clahe_layout = QHBoxLayout()
        self._clahe_enabled = QCheckBox("CLAHE")
        self._clahe_enabled.setChecked(True)
        self._clahe_enabled.setToolTip(
            "Contrast Limited Adaptive Histogram Equalization.\n"
            "Enhances local contrast to make features more visible.\n"
            "Recommended for images with uneven lighting."
        )
        self._clahe_enabled.stateChanged.connect(self._on_config_changed)
        clahe_layout.addWidget(self._clahe_enabled)
        
        clahe_layout.addWidget(QLabel("Clip:"))
        self._clahe_clip = QDoubleSpinBox()
        self._clahe_clip.setRange(1.0, 10.0)
        self._clahe_clip.setValue(2.0)
        self._clahe_clip.setSingleStep(0.5)
        self._clahe_clip.setToolTip(
            "Clip limit for contrast enhancement (1.0-10.0).\n"
            "Higher values = stronger contrast enhancement.\n"
            "Default: 2.0. Increase if image appears washed out."
        )
        self._clahe_clip.valueChanged.connect(self._on_config_changed)
        clahe_layout.addWidget(self._clahe_clip)
        
        clahe_layout.addStretch()
        layout.addLayout(clahe_layout)
        
        # Gaussian blur
        blur_layout = QHBoxLayout()
        self._blur_enabled = QCheckBox("Gaussian Blur")
        self._blur_enabled.setChecked(True)
        self._blur_enabled.setToolTip(
            "Applies slight smoothing to reduce noise.\n"
            "Helps feature detection by removing small artifacts.\n"
            "Recommended for noisy or grainy images."
        )
        self._blur_enabled.stateChanged.connect(self._on_config_changed)
        blur_layout.addWidget(self._blur_enabled)
        
        blur_layout.addWidget(QLabel("σ:"))
        self._blur_sigma = QDoubleSpinBox()
        self._blur_sigma.setRange(0.1, 5.0)
        self._blur_sigma.setValue(0.5)
        self._blur_sigma.setSingleStep(0.1)
        self._blur_sigma.setToolTip(
            "Blur strength (sigma, 0.1-5.0).\n"
            "Higher values = more smoothing.\n"
            "Default: 0.5. Keep low to preserve fine details."
        )
        self._blur_sigma.valueChanged.connect(self._on_config_changed)
        blur_layout.addWidget(self._blur_sigma)
        
        blur_layout.addStretch()
        layout.addLayout(blur_layout)
        
        return group
    
    def _create_multimodality_group(self) -> QGroupBox:
        """Create multimodality preprocessing options group."""
        group = QGroupBox("Multimodality Options")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        
        # Description label
        desc_label = QLabel("Adjust images for cross-modality registration:")
        desc_label.setStyleSheet("color: #a6adc8; font-size: 9pt;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Baseline (A) options
        baseline_label = QLabel("Baseline (A):")
        baseline_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(baseline_label)
        
        baseline_opts = QHBoxLayout()
        baseline_opts.addSpacing(16)
        
        self._baseline_grayscale = QCheckBox("Grayscale")
        self._baseline_grayscale.setChecked(False)
        self._baseline_grayscale.setToolTip(
            "Convert baseline image to grayscale before registration.\n"
            "Useful when baseline is a color fundus image and\n"
            "follow-up is a grayscale modality (FAF, IR, etc.)."
        )
        self._baseline_grayscale.stateChanged.connect(self._on_multimodal_changed)
        baseline_opts.addWidget(self._baseline_grayscale)
        
        self._baseline_invert = QCheckBox("Invert")
        self._baseline_invert.setChecked(False)
        self._baseline_invert.setToolTip(
            "Invert baseline image colors before registration.\n"
            "Useful when vessel appearance differs between modalities:\n"
            "• FAF: vessels appear dark\n"
            "• IR: vessels appear bright\n"
            "Invert to match vessel polarity for better alignment."
        )
        self._baseline_invert.stateChanged.connect(self._on_multimodal_changed)
        baseline_opts.addWidget(self._baseline_invert)
        
        baseline_opts.addStretch()
        layout.addLayout(baseline_opts)
        
        # Baseline contrast/brightness
        baseline_cb_layout = QHBoxLayout()
        baseline_cb_layout.addSpacing(16)
        baseline_cb_layout.addWidget(QLabel("Contrast:"))
        self._baseline_contrast = QSlider(Qt.Orientation.Horizontal)
        self._baseline_contrast.setRange(50, 200)
        self._baseline_contrast.setValue(100)
        self._baseline_contrast.setToolTip("Adjust baseline contrast (50-200%)\nDefault: 100%")
        self._baseline_contrast.valueChanged.connect(self._on_multimodal_changed)
        baseline_cb_layout.addWidget(self._baseline_contrast)
        self._baseline_contrast_label = QLabel("100%")
        self._baseline_contrast_label.setFixedWidth(35)
        baseline_cb_layout.addWidget(self._baseline_contrast_label)
        layout.addLayout(baseline_cb_layout)
        
        baseline_br_layout = QHBoxLayout()
        baseline_br_layout.addSpacing(16)
        baseline_br_layout.addWidget(QLabel("Brightness:"))
        self._baseline_brightness = QSlider(Qt.Orientation.Horizontal)
        self._baseline_brightness.setRange(-100, 100)
        self._baseline_brightness.setValue(0)
        self._baseline_brightness.setToolTip("Adjust baseline brightness (-100 to +100)\nDefault: 0")
        self._baseline_brightness.valueChanged.connect(self._on_multimodal_changed)
        baseline_br_layout.addWidget(self._baseline_brightness)
        self._baseline_brightness_label = QLabel("0")
        self._baseline_brightness_label.setFixedWidth(35)
        baseline_br_layout.addWidget(self._baseline_brightness_label)
        layout.addLayout(baseline_br_layout)
        
        # Baseline CLAHE (for multimodal registration)
        baseline_clahe_layout = QHBoxLayout()
        baseline_clahe_layout.addSpacing(16)
        self._baseline_clahe = QCheckBox("CLAHE")
        self._baseline_clahe.setChecked(False)
        self._baseline_clahe.setToolTip(
            "Apply Contrast Limited Adaptive Histogram Equalization.\n"
            "Recommended for multi-modal registration (IR↔FAF).\n"
            "Enhances local contrast while preventing over-amplification."
        )
        self._baseline_clahe.stateChanged.connect(self._on_multimodal_changed)
        baseline_clahe_layout.addWidget(self._baseline_clahe)
        baseline_clahe_layout.addStretch()
        layout.addLayout(baseline_clahe_layout)
        
        # Follow-up (B) options
        followup_label = QLabel("Follow-up (B):")
        followup_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(followup_label)
        
        followup_opts = QHBoxLayout()
        followup_opts.addSpacing(16)
        
        self._followup_grayscale = QCheckBox("Grayscale")
        self._followup_grayscale.setChecked(False)
        self._followup_grayscale.setToolTip(
            "Convert follow-up image to grayscale before registration.\n"
            "Useful when follow-up is a color fundus image and\n"
            "baseline is a grayscale modality (FAF, IR, etc.)."
        )
        self._followup_grayscale.stateChanged.connect(self._on_multimodal_changed)
        followup_opts.addWidget(self._followup_grayscale)
        
        self._followup_invert = QCheckBox("Invert")
        self._followup_invert.setChecked(False)
        self._followup_invert.setToolTip(
            "Invert follow-up image colors before registration.\n"
            "Useful when vessel appearance differs between modalities:\n"
            "• FAF: vessels appear dark\n"
            "• IR: vessels appear bright\n"
            "Invert to match vessel polarity for better alignment."
        )
        self._followup_invert.stateChanged.connect(self._on_multimodal_changed)
        followup_opts.addWidget(self._followup_invert)
        
        followup_opts.addStretch()
        layout.addLayout(followup_opts)
        
        # Follow-up contrast/brightness
        followup_cb_layout = QHBoxLayout()
        followup_cb_layout.addSpacing(16)
        followup_cb_layout.addWidget(QLabel("Contrast:"))
        self._followup_contrast = QSlider(Qt.Orientation.Horizontal)
        self._followup_contrast.setRange(50, 200)
        self._followup_contrast.setValue(100)
        self._followup_contrast.setToolTip("Adjust follow-up contrast (50-200%)\nDefault: 100%")
        self._followup_contrast.valueChanged.connect(self._on_multimodal_changed)
        followup_cb_layout.addWidget(self._followup_contrast)
        self._followup_contrast_label = QLabel("100%")
        self._followup_contrast_label.setFixedWidth(35)
        followup_cb_layout.addWidget(self._followup_contrast_label)
        layout.addLayout(followup_cb_layout)
        
        followup_br_layout = QHBoxLayout()
        followup_br_layout.addSpacing(16)
        followup_br_layout.addWidget(QLabel("Brightness:"))
        self._followup_brightness = QSlider(Qt.Orientation.Horizontal)
        self._followup_brightness.setRange(-100, 100)
        self._followup_brightness.setValue(0)
        self._followup_brightness.setToolTip("Adjust follow-up brightness (-100 to +100)\nDefault: 0")
        self._followup_brightness.valueChanged.connect(self._on_multimodal_changed)
        followup_br_layout.addWidget(self._followup_brightness)
        self._followup_brightness_label = QLabel("0")
        self._followup_brightness_label.setFixedWidth(35)
        followup_br_layout.addWidget(self._followup_brightness_label)
        layout.addLayout(followup_br_layout)
        
        # Follow-up CLAHE (for multimodal registration)
        followup_clahe_layout = QHBoxLayout()
        followup_clahe_layout.addSpacing(16)
        self._followup_clahe = QCheckBox("CLAHE")
        self._followup_clahe.setChecked(False)
        self._followup_clahe.setToolTip(
            "Apply Contrast Limited Adaptive Histogram Equalization.\n"
            "Recommended for multi-modal registration (IR↔FAF).\n"
            "Enhances local contrast while preventing over-amplification."
        )
        self._followup_clahe.stateChanged.connect(self._on_multimodal_changed)
        followup_clahe_layout.addWidget(self._followup_clahe)
        followup_clahe_layout.addStretch()
        layout.addLayout(followup_clahe_layout)
        
        # Reset button
        reset_btn = QPushButton("Reset All")
        reset_btn.setToolTip("Reset all multimodality options to defaults")
        reset_btn.clicked.connect(self._reset_multimodal_options)
        layout.addWidget(reset_btn)
        
        # Note about display
        note_label = QLabel("Note: Preview shows adjustments. Final results use original colors.")
        note_label.setStyleSheet("color: #6c7086; font-size: 8pt; font-style: italic;")
        note_label.setWordWrap(True)
        layout.addWidget(note_label)
        
        return group
    
    def _create_registration_group(self) -> QGroupBox:
        """Create registration settings group."""
        group = QGroupBox("Alignment Options")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        
        # Motion model
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Motion Model:"))
        
        self._motion_model = QComboBox()
        self._motion_model.addItem("Translation", MotionModel.TRANSLATION)
        self._motion_model.addItem("Euclidean", MotionModel.EUCLIDEAN)
        self._motion_model.addItem("Affine", MotionModel.AFFINE)
        self._motion_model.addItem("Homography", MotionModel.HOMOGRAPHY)
        self._motion_model.setCurrentIndex(2)  # Affine default
        self._motion_model.setToolTip(
            "Type of geometric transformation to estimate:\n"
            "• Translation: Shift only (X, Y movement)\n"
            "• Euclidean: Shift + rotation (rigid body)\n"
            "• Affine: Shift + rotation + scaling + shear (recommended)\n"
            "• Homography: Full perspective transform (for tilted images)"
        )
        self._motion_model.currentIndexChanged.connect(self._on_config_changed)
        model_layout.addWidget(self._motion_model, 1)
        
        layout.addLayout(model_layout)
        
        # Alignment method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        
        self._coarse_method = QComboBox()
        self._coarse_method.addItem("AKAZE (Robust)", "akaze")
        self._coarse_method.addItem("ORB (Fast)", "orb")
        self._coarse_method.addItem("SIFT (Classic)", "sift")
        self._coarse_method.setCurrentIndex(0)  # AKAZE default
        self._coarse_method.setToolTip(
            "Feature detection and matching algorithm:\n"
            "• AKAZE: Best for most ophthalmic images (recommended)\n"
            "• ORB: Fastest, good for real-time processing\n"
            "• SIFT: Classic algorithm, very accurate\n\n"
            "Tip: For multi-modal (IR↔FAF), use CLAHE preprocessing"
        )
        self._coarse_method.currentIndexChanged.connect(self._on_method_changed)
        method_layout.addWidget(self._coarse_method, 1)
        
        layout.addLayout(method_layout)
        
        # Method-specific options container
        self._method_options_container = QWidget()
        method_options_layout = QVBoxLayout(self._method_options_container)
        method_options_layout.setContentsMargins(0, 0, 0, 0)
        method_options_layout.setSpacing(4)
        
        # AKAZE options
        self._akaze_options = QWidget()
        akaze_layout = QVBoxLayout(self._akaze_options)
        akaze_layout.setContentsMargins(0, 0, 0, 0)
        akaze_layout.setSpacing(4)
        
        akaze_thresh_layout = QHBoxLayout()
        akaze_thresh_layout.addSpacing(24)
        self._akaze_thresh_label = QLabel("Threshold:")
        akaze_thresh_layout.addWidget(self._akaze_thresh_label)
        self._akaze_threshold = QDoubleSpinBox()
        self._akaze_threshold.setRange(0.0001, 0.01)
        self._akaze_threshold.setValue(0.001)
        self._akaze_threshold.setSingleStep(0.0005)
        self._akaze_threshold.setDecimals(4)
        self._akaze_threshold.setToolTip(
            "AKAZE detector threshold (0.0001-0.01).\n"
            "Lower = more features detected.\n"
            "Higher = fewer but stronger features.\n"
            "Default: 0.001"
        )
        self._akaze_threshold.valueChanged.connect(self._on_config_changed)
        akaze_thresh_layout.addWidget(self._akaze_threshold)
        akaze_thresh_layout.addStretch()
        akaze_layout.addLayout(akaze_thresh_layout)
        
        akaze_ratio_layout = QHBoxLayout()
        akaze_ratio_layout.addSpacing(24)
        akaze_ratio_layout.addWidget(QLabel("Match Ratio:"))
        self._akaze_ratio = QDoubleSpinBox()
        self._akaze_ratio.setRange(0.5, 0.95)
        self._akaze_ratio.setValue(0.75)
        self._akaze_ratio.setSingleStep(0.05)
        self._akaze_ratio.setToolTip(
            "Feature matching strictness (0.5-0.95).\n"
            "Lower = stricter matching, fewer but better matches.\n"
            "Higher = more matches but may include errors.\n"
            "Default: 0.75"
        )
        self._akaze_ratio.valueChanged.connect(self._on_config_changed)
        akaze_ratio_layout.addWidget(self._akaze_ratio)
        akaze_ratio_layout.addStretch()
        akaze_layout.addLayout(akaze_ratio_layout)
        
        method_options_layout.addWidget(self._akaze_options)
        
        # ORB options
        self._orb_options = QWidget()
        orb_layout = QVBoxLayout(self._orb_options)
        orb_layout.setContentsMargins(0, 0, 0, 0)
        orb_layout.setSpacing(4)
        
        orb_features_layout = QHBoxLayout()
        orb_features_layout.addSpacing(24)
        orb_features_layout.addWidget(QLabel("Max Features:"))
        self._orb_features = QSpinBox()
        self._orb_features.setRange(100, 10000)
        self._orb_features.setValue(2000)
        self._orb_features.setSingleStep(500)
        self._orb_features.setToolTip(
            "Maximum number of ORB features to detect (100-10000).\n"
            "More features = better accuracy but slower.\n"
            "Default: 2000"
        )
        self._orb_features.valueChanged.connect(self._on_config_changed)
        orb_features_layout.addWidget(self._orb_features)
        orb_features_layout.addStretch()
        orb_layout.addLayout(orb_features_layout)
        
        orb_ratio_layout = QHBoxLayout()
        orb_ratio_layout.addSpacing(24)
        orb_ratio_layout.addWidget(QLabel("Match Ratio:"))
        self._orb_ratio = QDoubleSpinBox()
        self._orb_ratio.setRange(0.5, 0.95)
        self._orb_ratio.setValue(0.75)
        self._orb_ratio.setSingleStep(0.05)
        self._orb_ratio.setToolTip(
            "Feature matching strictness (0.5-0.95).\n"
            "Lower = stricter matching, fewer but better matches.\n"
            "Default: 0.75"
        )
        self._orb_ratio.valueChanged.connect(self._on_config_changed)
        orb_ratio_layout.addWidget(self._orb_ratio)
        orb_ratio_layout.addStretch()
        orb_layout.addLayout(orb_ratio_layout)
        
        self._orb_options.setVisible(False)
        method_options_layout.addWidget(self._orb_options)
        
        # SIFT options
        self._sift_options = QWidget()
        sift_layout = QVBoxLayout(self._sift_options)
        sift_layout.setContentsMargins(0, 0, 0, 0)
        sift_layout.setSpacing(4)
        
        sift_features_layout = QHBoxLayout()
        sift_features_layout.addSpacing(24)
        sift_features_layout.addWidget(QLabel("Max Features:"))
        self._sift_features = QSpinBox()
        self._sift_features.setRange(100, 20000)
        self._sift_features.setValue(5000)
        self._sift_features.setSingleStep(500)
        self._sift_features.setToolTip(
            "Maximum number of SIFT features to detect (100-20000).\n"
            "More features = better accuracy but slower.\n"
            "Default: 5000. Increase for complex images."
        )
        self._sift_features.valueChanged.connect(self._on_config_changed)
        sift_features_layout.addWidget(self._sift_features)
        sift_features_layout.addStretch()
        sift_layout.addLayout(sift_features_layout)
        
        sift_ratio_layout = QHBoxLayout()
        sift_ratio_layout.addSpacing(24)
        sift_ratio_layout.addWidget(QLabel("Match Ratio:"))
        self._sift_ratio = QDoubleSpinBox()
        self._sift_ratio.setRange(0.5, 0.95)
        self._sift_ratio.setValue(0.75)
        self._sift_ratio.setSingleStep(0.05)
        self._sift_ratio.setToolTip(
            "Feature matching strictness (0.5-0.95).\n"
            "Lower = stricter matching, fewer but better matches.\n"
            "Default: 0.75"
        )
        self._sift_ratio.valueChanged.connect(self._on_config_changed)
        sift_ratio_layout.addWidget(self._sift_ratio)
        sift_ratio_layout.addStretch()
        sift_layout.addLayout(sift_ratio_layout)
        
        self._sift_options.setVisible(False)
        method_options_layout.addWidget(self._sift_options)
        
        layout.addWidget(self._method_options_container)
        
        # Keep legacy match_ratio for compatibility
        self._match_ratio = self._akaze_ratio
        
        return group
    
    def _create_validation_group(self) -> QGroupBox:
        """Create validation settings group."""
        group = QGroupBox("Validation")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        
        # Enable validation
        self._validation_enabled = QCheckBox("Validate Transform")
        self._validation_enabled.setChecked(True)
        self._validation_enabled.setToolTip(
            "Check if the computed transform is reasonable.\n"
            "Rejects transforms with excessive rotation or translation.\n"
            "Recommended to keep enabled to avoid bad registrations."
        )
        self._validation_enabled.stateChanged.connect(self._on_config_changed)
        layout.addWidget(self._validation_enabled)
        
        # Max translation
        trans_layout = QHBoxLayout()
        trans_layout.addSpacing(24)
        trans_layout.addWidget(QLabel("Max Translation (px):"))
        
        self._max_translation = QSpinBox()
        self._max_translation.setRange(10, 2000)
        self._max_translation.setValue(500)
        self._max_translation.setSingleStep(50)
        self._max_translation.setToolTip(
            "Maximum allowed shift in pixels (10-2000).\n"
            "Transforms exceeding this will be rejected.\n"
            "Default: 500. Increase for images with large shifts."
        )
        trans_layout.addWidget(self._max_translation)
        
        trans_layout.addStretch()
        layout.addLayout(trans_layout)
        
        # Max rotation
        rot_layout = QHBoxLayout()
        rot_layout.addSpacing(24)
        rot_layout.addWidget(QLabel("Max Rotation (°):"))
        
        self._max_rotation = QSpinBox()
        self._max_rotation.setRange(1, 180)
        self._max_rotation.setValue(45)
        self._max_rotation.setSingleStep(5)
        self._max_rotation.setToolTip(
            "Maximum allowed rotation in degrees (1-180).\n"
            "Transforms exceeding this will be rejected.\n"
            "Default: 45°. Increase for rotated images."
        )
        rot_layout.addWidget(self._max_rotation)
        
        rot_layout.addStretch()
        layout.addLayout(rot_layout)
        
        return group
    
    def _on_config_changed(self) -> None:
        """Handle configuration change."""
        self.configChanged.emit()
    
    def get_registration_config(self) -> RegistrationConfig:
        """Get the current registration configuration."""
        method = self._coarse_method.currentData()
        
        # Get method-specific settings
        if method == "akaze":
            n_features = 5000  # AKAZE doesn't use n_features directly
            match_ratio = self._akaze_ratio.value()
        elif method == "orb":
            n_features = self._orb_features.value()
            match_ratio = self._orb_ratio.value()
        elif method == "sift":
            n_features = self._sift_features.value()
            match_ratio = self._sift_ratio.value()
        else:
            n_features = 5000
            match_ratio = 0.75
        
        return RegistrationConfig(
            motion_model=self._motion_model.currentData(),
            use_coarse_alignment=True,
            use_fine_alignment=False,
            sift_n_features=n_features,
            match_ratio_threshold=match_ratio,
            ecc_max_iterations=100,
            validate_transform=self._validation_enabled.isChecked(),
            max_translation_pixels=float(self._max_translation.value()),
            max_rotation_degrees=float(self._max_rotation.value()),
        )
    
    def get_coarse_method(self) -> str:
        """Get the selected coarse alignment method."""
        return self._coarse_method.currentData()
    
    def get_n_features(self) -> int:
        """Get the max features setting based on current method."""
        method = self._coarse_method.currentData()
        if method == "orb":
            return self._orb_features.value()
        elif method == "sift":
            return self._sift_features.value()
        return 5000  # Default for AKAZE
    
    def get_akaze_threshold(self) -> float:
        """Get AKAZE threshold setting."""
        return self._akaze_threshold.value()
    
    def get_multimodality_options(self) -> dict:
        """Get multimodality preprocessing options."""
        return {
            "baseline_grayscale": self._baseline_grayscale.isChecked(),
            "baseline_invert": self._baseline_invert.isChecked(),
            "baseline_contrast": self._baseline_contrast.value() / 100.0,
            "baseline_brightness": self._baseline_brightness.value(),
            "baseline_clahe": self._baseline_clahe.isChecked(),
            "followup_grayscale": self._followup_grayscale.isChecked(),
            "followup_invert": self._followup_invert.isChecked(),
            "followup_contrast": self._followup_contrast.value() / 100.0,
            "followup_brightness": self._followup_brightness.value(),
            "followup_clahe": self._followup_clahe.isChecked(),
        }
    
    def _on_multimodal_changed(self) -> None:
        """Handle multimodality option change - update labels and emit signal."""
        # Update labels
        self._baseline_contrast_label.setText(f"{self._baseline_contrast.value()}%")
        self._baseline_brightness_label.setText(str(self._baseline_brightness.value()))
        self._followup_contrast_label.setText(f"{self._followup_contrast.value()}%")
        self._followup_brightness_label.setText(str(self._followup_brightness.value()))
        
        # Emit signal for live preview
        self.multimodalOptionsChanged.emit(self.get_multimodality_options())
        self._on_config_changed()
    
    def _reset_multimodal_options(self) -> None:
        """Reset all multimodality options to defaults."""
        self._baseline_grayscale.setChecked(False)
        self._baseline_invert.setChecked(False)
        self._baseline_contrast.setValue(100)
        self._baseline_brightness.setValue(0)
        self._baseline_clahe.setChecked(False)
        self._followup_grayscale.setChecked(False)
        self._followup_invert.setChecked(False)
        self._followup_contrast.setValue(100)
        self._followup_brightness.setValue(0)
        self._followup_clahe.setChecked(False)
    
    def get_preprocessing_config(self) -> Optional[PreprocessingConfig]:
        """Get the current preprocessing configuration."""
        if not self._preproc_enabled.isChecked():
            return None
        
        steps = [PreprocessingStep.GRAYSCALE]
        
        if self._clahe_enabled.isChecked():
            steps.append(PreprocessingStep.CLAHE)
        
        if self._blur_enabled.isChecked():
            steps.append(PreprocessingStep.GAUSSIAN_BLUR)
        
        return PreprocessingConfig(
            steps=steps,
            clahe_clip_limit=self._clahe_clip.value(),
            gaussian_sigma=self._blur_sigma.value(),
        )
    
    def set_status(self, message: str) -> None:
        """Set status message."""
        self._status_label.setText(message)
    
    def set_progress(self, value: int, visible: bool = True) -> None:
        """Set progress bar value and visibility."""
        self._progress_bar.setVisible(visible)
        self._progress_bar.setValue(value)
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the register button."""
        self._register_btn.setEnabled(enabled)
    
    def _on_show_reg_points_changed(self, state: int) -> None:
        """Handle registration points toggle change."""
        self.showRegistrationPointsChanged.emit(state == Qt.CheckState.Checked.value)
    
    def _on_show_vessel_contours_changed(self, state: int) -> None:
        """Handle vessel contours toggle change."""
        self.showVesselContoursChanged.emit(state == Qt.CheckState.Checked.value)
    
    def _on_method_changed(self, index: int) -> None:
        """Handle alignment method change - show/hide relevant options."""
        method = self._coarse_method.currentData()
        
        # Hide all method options
        self._akaze_options.setVisible(False)
        self._orb_options.setVisible(False)
        self._sift_options.setVisible(False)
        
        # Show relevant options
        if method == "akaze":
            self._akaze_options.setVisible(True)
            self._match_ratio = self._akaze_ratio
        elif method == "orb":
            self._orb_options.setVisible(True)
            self._match_ratio = self._orb_ratio
        elif method == "sift":
            self._sift_options.setVisible(True)
            self._match_ratio = self._sift_ratio
        
        self._on_config_changed()


class ComparisonMode:
    """Comparison mode enum values."""
    SIDE_BY_SIDE = "Side by Side"
    OVERLAY = "Overlay"
    DIFFERENCE = "Difference"
    CHECKERBOARD = "Checkerboard"
    SPLIT = "Split View"


class ResultsPanel(QWidget):
    """
    Panel for displaying registration results.
    
    Shows metrics, transform parameters, and quality assessment.
    Also includes comparison mode controls.
    """
    
    exportClicked = pyqtSignal()
    comparisonModeChanged = pyqtSignal(str)
    overlayAlphaChanged = pyqtSignal(int)
    checkerSizeChanged = pyqtSignal(int)
    nativeColorChanged = pyqtSignal(bool)
    fitRequested = pyqtSignal()
    zoom100Requested = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.setFixedWidth(320)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the results UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)
        
        # Comparison Mode group (at top)
        comparison_group = self._create_comparison_group()
        layout.addWidget(comparison_group)
        
        # Title
        title = QLabel("Registration Results")
        title.setObjectName("titleLabel")
        layout.addWidget(title)
        
        # Quality indicator
        quality_group = QGroupBox("Quality")
        quality_layout = QVBoxLayout(quality_group)
        
        self._quality_label = QLabel("--")
        self._quality_label.setStyleSheet("""
            font-size: 18pt;
            font-weight: bold;
            color: #a6adc8;
            padding: 12px;
        """)
        self._quality_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        quality_layout.addWidget(self._quality_label)
        
        layout.addWidget(quality_group)
        
        # Transform parameters
        transform_group = QGroupBox("Transform")
        transform_layout = QVBoxLayout(transform_group)
        
        self._translation_label = QLabel("Translation: -- , --")
        transform_layout.addWidget(self._translation_label)
        
        self._rotation_label = QLabel("Rotation: --°")
        transform_layout.addWidget(self._rotation_label)
        
        self._scale_label = QLabel("Scale: -- , --")
        transform_layout.addWidget(self._scale_label)
        
        layout.addWidget(transform_group)
        
        # Metrics
        metrics_group = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self._inliers_label = QLabel("Inliers: --")
        metrics_layout.addWidget(self._inliers_label)
        
        self._overlap_label = QLabel("Overlap: --")
        metrics_layout.addWidget(self._overlap_label)
        
        self._time_label = QLabel("Time: --")
        metrics_layout.addWidget(self._time_label)
        
        layout.addWidget(metrics_group)
        
        # Warnings
        self._warnings_group = QGroupBox("Warnings")
        self._warnings_layout = QVBoxLayout(self._warnings_group)
        self._warnings_label = QLabel("None")
        self._warnings_label.setWordWrap(True)
        self._warnings_label.setStyleSheet("color: #f9e2af;")
        self._warnings_layout.addWidget(self._warnings_label)
        self._warnings_group.setVisible(False)
        layout.addWidget(self._warnings_group)
        
        layout.addStretch()
        
        # Export button
        self._export_btn = QPushButton("📁  Export Results")
        self._export_btn.setObjectName("secondaryButton")
        self._export_btn.setMinimumHeight(40)
        self._export_btn.clicked.connect(self.exportClicked.emit)
        self._export_btn.setEnabled(False)
        layout.addWidget(self._export_btn)
    
    def _create_comparison_group(self) -> QGroupBox:
        """Create comparison mode controls group."""
        group = QGroupBox("Comparison View")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        
        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        
        self._mode_combo = QComboBox()
        self._mode_combo.addItem(ComparisonMode.SIDE_BY_SIDE)
        self._mode_combo.addItem(ComparisonMode.OVERLAY)
        self._mode_combo.addItem(ComparisonMode.DIFFERENCE)
        self._mode_combo.addItem(ComparisonMode.CHECKERBOARD)
        self._mode_combo.addItem(ComparisonMode.SPLIT)
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode_combo, 1)
        
        layout.addLayout(mode_layout)
        
        # Overlay alpha slider
        self._overlay_container = QWidget()
        overlay_layout = QVBoxLayout(self._overlay_container)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        
        alpha_label_layout = QHBoxLayout()
        alpha_label_layout.addWidget(QLabel("Baseline"))
        alpha_label_layout.addStretch()
        alpha_label_layout.addWidget(QLabel("Registered"))
        overlay_layout.addLayout(alpha_label_layout)
        
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, 100)
        self._alpha_slider.setValue(50)
        self._alpha_slider.valueChanged.connect(self._on_alpha_changed)
        overlay_layout.addWidget(self._alpha_slider)
        
        # Native color toggle
        color_toggle_layout = QHBoxLayout()
        color_toggle_layout.addWidget(QLabel("Native Colors:"))
        self._native_color_checkbox = QCheckBox()
        self._native_color_checkbox.setChecked(False)
        self._native_color_checkbox.stateChanged.connect(self._on_native_color_changed)
        color_toggle_layout.addWidget(self._native_color_checkbox)
        color_toggle_layout.addStretch()
        overlay_layout.addLayout(color_toggle_layout)
        
        self._overlay_container.setVisible(False)
        layout.addWidget(self._overlay_container)
        
        # Checkerboard size selector
        self._checker_container = QWidget()
        checker_layout = QHBoxLayout(self._checker_container)
        checker_layout.setContentsMargins(0, 0, 0, 0)
        
        checker_layout.addWidget(QLabel("Grid Size:"))
        self._checker_combo = QComboBox()
        for size in [25, 50, 75, 100, 150]:
            self._checker_combo.addItem(f"{size}px", size)
        self._checker_combo.setCurrentIndex(1)
        self._checker_combo.currentIndexChanged.connect(self._on_checker_changed)
        checker_layout.addWidget(self._checker_combo, 1)
        
        self._checker_container.setVisible(False)
        layout.addWidget(self._checker_container)
        
        # View control buttons
        view_layout = QHBoxLayout()
        view_layout.setSpacing(8)
        
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self.fitRequested.emit)
        view_layout.addWidget(fit_btn)
        
        zoom_100_btn = QPushButton("1:1")
        zoom_100_btn.clicked.connect(self.zoom100Requested.emit)
        view_layout.addWidget(zoom_100_btn)
        
        layout.addLayout(view_layout)
        
        return group
    
    def _on_mode_changed(self, mode: str) -> None:
        """Handle comparison mode change."""
        # Show/hide mode-specific controls
        self._overlay_container.setVisible(mode == ComparisonMode.OVERLAY)
        self._checker_container.setVisible(mode == ComparisonMode.CHECKERBOARD)
        
        self.comparisonModeChanged.emit(mode)
    
    def _on_alpha_changed(self, value: int) -> None:
        """Handle overlay alpha change."""
        self.overlayAlphaChanged.emit(value)
    
    def _on_native_color_changed(self, state: int) -> None:
        """Handle native color toggle change."""
        self.nativeColorChanged.emit(state == Qt.CheckState.Checked.value)
    
    def _on_checker_changed(self, index: int) -> None:
        """Handle checkerboard size change."""
        size = self._checker_combo.currentData()
        self.checkerSizeChanged.emit(size)
    
    def set_result(self, result) -> None:
        """
        Update the panel with registration results.
        
        Args:
            result: TransformResult object
        """
        # Get alignment method from quality metrics
        method = result.quality_metrics.get("alignment_method", "unknown")
        
        # Check for alignment failure first
        alignment_failed = False
        if result.warnings:
            for warning in result.warnings:
                if "failed" in warning.lower() or "insufficient" in warning.lower():
                    alignment_failed = True
                    break
        
        # Determine quality based on method-specific metrics
        overlap = result.quality_metrics.get("overlap_ratio", 0)
        confidence_score = 0
        
        if alignment_failed:
            # Registration failed
            quality = "Failed"
            color = "#ff0000"
            confidence_score = 0
        elif result.feature_match_result:
            # For feature-based methods, use inlier count
            inliers = result.feature_match_result.num_inliers
            reproj_error = result.quality_metrics.get("coarse_reproj_error_mean", 0)
            
            # Check for very low inliers (likely failed)
            if inliers < 4:
                quality = "Failed"
                color = "#ff0000"
                confidence_score = 0
            else:
                # Calculate confidence score (0-100%)
                # Based on inliers and reprojection error
                inlier_score = min(inliers / 50.0, 1.0) * 60  # Up to 60% from inliers
                error_score = max(0, (10 - reproj_error) / 10.0) * 40  # Up to 40% from low error
                confidence_score = int(inlier_score + error_score)
                
                if inliers >= 50 and reproj_error < 3.0:
                    quality = "Excellent"
                    color = "#a6e3a1"
                elif inliers >= 20 and reproj_error < 5.0:
                    quality = "Good"
                    color = "#94e2d5"
                elif inliers >= 10:
                    quality = "Fair"
                    color = "#f9e2af"
                else:
                    quality = "Poor"
                    color = "#f38ba8"
        else:
            # No feature match result - likely failed or identity transform
            quality = "Failed"
            color = "#ff0000"
            confidence_score = 0
        
        self._quality_label.setText(f"{quality}\n{confidence_score}%")
        self._quality_label.setStyleSheet(f"""
            font-size: 18pt;
            font-weight: bold;
            color: {color};
            padding: 12px;
        """)
        
        
        # Transform
        tx, ty = result.translation
        self._translation_label.setText(f"Translation: {tx:.2f} , {ty:.2f} px")
        
        rot = result.rotation_degrees or 0
        self._rotation_label.setText(f"Rotation: {rot:.3f}°")
        
        sx, sy = result.scale_factors
        self._scale_label.setText(f"Scale: {sx:.4f} , {sy:.4f}")
        
        # Metrics
        if result.feature_match_result:
            total_matches = result.quality_metrics.get("coarse_num_matches", 0)
            self._inliers_label.setText(f"Matches: {total_matches}")
        else:
            iterations = result.quality_metrics.get("iterations", 0)
            self._inliers_label.setText(f"Iterations: {iterations}")
        
        self._overlap_label.setText(f"Overlap: {overlap:.1%}")
        
        self._time_label.setText(f"Time: {result.registration_time_ms:.1f} ms")
        
        # Warnings
        if result.warnings:
            self._warnings_group.setVisible(True)
            warnings_text = "\n".join(f"• {w}" for w in result.warnings)
            self._warnings_label.setText(warnings_text)
            # Use bright red for failed/poor quality registrations or any failure message
            if quality == "Failed" or quality == "Poor" or confidence_score < 30 or alignment_failed:
                self._warnings_label.setStyleSheet("color: #ff0000; font-weight: bold;")
            else:
                self._warnings_label.setStyleSheet("color: #f9e2af;")
        else:
            self._warnings_group.setVisible(False)
        
        # Enable export
        self._export_btn.setEnabled(True)
    
    def clear(self) -> None:
        """Clear the results panel."""
        self._quality_label.setText("--")
        self._quality_label.setStyleSheet("""
            font-size: 18pt;
            font-weight: bold;
            color: #a6adc8;
            padding: 12px;
        """)
        self._translation_label.setText("Translation: -- , --")
        self._rotation_label.setText("Rotation: --°")
        self._scale_label.setText("Scale: -- , --")
        self._inliers_label.setText("Inliers: --")
        self._overlap_label.setText("Overlap: --")
        self._time_label.setText("Time: --")
        self._warnings_group.setVisible(False)
        self._export_btn.setEnabled(False)
    
    def set_comparison_mode(self, mode: str) -> None:
        """Set the comparison mode dropdown without triggering signal."""
        # Block signals to avoid circular updates
        self._mode_combo.blockSignals(True)
        index = self._mode_combo.findText(mode)
        if index >= 0:
            self._mode_combo.setCurrentIndex(index)
        self._mode_combo.blockSignals(False)
        
        # Manually update visibility of mode-specific controls
        self._overlay_container.setVisible(mode == ComparisonMode.OVERLAY)
        self._checker_container.setVisible(mode == ComparisonMode.CHECKERBOARD)
