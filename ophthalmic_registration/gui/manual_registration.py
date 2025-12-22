"""
Manual 3-point registration dialog.

Allows users to manually select corresponding points in baseline and
follow-up images to compute an affine transformation.
"""

import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QFrame, QGroupBox, QMessageBox, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QPen, QColor, QBrush

from ophthalmic_registration.core.image_data import ImageData, TransformResult, MotionModel
from ophthalmic_registration.gui.image_viewer import ImageViewer

logger = logging.getLogger(__name__)


@dataclass
class PointPair:
    """A pair of corresponding points."""
    baseline_point: Optional[QPointF] = None
    followup_point: Optional[QPointF] = None
    color: QColor = field(default_factory=lambda: QColor(255, 0, 0))


class PointSelectionViewer(ImageViewer):
    """
    Image viewer with point selection capability.
    
    Allows clicking to place numbered markers on the image.
    """
    
    pointClicked = pyqtSignal(QPointF)  # Emits scene coordinates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._points: List[Tuple[QPointF, QColor]] = []
        self._point_items = []
        self._label_items = []
        self._selection_enabled = True
    
    def set_selection_enabled(self, enabled: bool) -> None:
        """Enable or disable point selection."""
        self._selection_enabled = enabled
    
    def mousePressEvent(self, event):
        """Handle mouse press for point selection."""
        if self._selection_enabled and event.button() == Qt.MouseButton.LeftButton:
            # Get scene position
            scene_pos = self.mapToScene(event.pos())
            
            # Check if within image bounds
            if self._pixmap_item and self._pixmap_item.pixmap():
                pixmap = self._pixmap_item.pixmap()
                if 0 <= scene_pos.x() < pixmap.width() and 0 <= scene_pos.y() < pixmap.height():
                    self.pointClicked.emit(scene_pos)
                    return
        
        super().mousePressEvent(event)
    
    def add_point(self, point: QPointF, color: QColor, label: str) -> None:
        """Add a point marker to the viewer."""
        from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem
        from PyQt6.QtGui import QFont
        
        # Create circle marker
        radius = 8
        ellipse = self._scene.addEllipse(
            point.x() - radius, point.y() - radius,
            radius * 2, radius * 2,
            QPen(color, 2),
            QBrush(QColor(color.red(), color.green(), color.blue(), 100))
        )
        self._point_items.append(ellipse)
        
        # Create label
        text = self._scene.addText(label, QFont("Arial", 12, QFont.Weight.Bold))
        text.setDefaultTextColor(color)
        text.setPos(point.x() + radius + 2, point.y() - radius)
        self._label_items.append(text)
        
        self._points.append((point, color))
    
    def clear_points(self) -> None:
        """Clear all point markers."""
        for item in self._point_items:
            self._scene.removeItem(item)
        for item in self._label_items:
            self._scene.removeItem(item)
        self._point_items.clear()
        self._label_items.clear()
        self._points.clear()
    
    def get_points(self) -> List[QPointF]:
        """Get all selected points."""
        return [p for p, _ in self._points]


class ManualRegistrationDialog(QDialog):
    """
    Dialog for manual 3-point registration.
    
    Users click on 3 corresponding features in baseline and follow-up
    images to compute an affine transformation.
    """
    
    registrationComplete = pyqtSignal(object, object)  # result, registered_image
    
    POINT_COLORS = [
        QColor(255, 0, 0),    # Red
        QColor(0, 255, 0),    # Green
        QColor(0, 0, 255),    # Blue
    ]
    
    def __init__(
        self,
        baseline: ImageData,
        followup: ImageData,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._baseline = baseline
        self._followup = followup
        self._point_pairs: List[PointPair] = []
        self._current_point_index = 0
        self._selecting_baseline = True
        
        self.setWindowTitle("Manual 3-Point Registration")
        self.setMinimumSize(1200, 700)
        
        self._setup_ui()
        self._load_images()
        self._update_instructions()
    
    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Instructions
        self._instructions = QLabel()
        self._instructions.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                padding: 10px;
                background-color: #1e1e2e;
                border-radius: 5px;
            }
        """)
        self._instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._instructions)
        
        # Image viewers
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Baseline panel
        baseline_frame = QFrame()
        baseline_layout = QVBoxLayout(baseline_frame)
        baseline_layout.setContentsMargins(0, 0, 0, 0)
        
        baseline_label = QLabel("Baseline Image")
        baseline_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        baseline_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        baseline_layout.addWidget(baseline_label)
        
        self._baseline_viewer = PointSelectionViewer()
        self._baseline_viewer.pointClicked.connect(self._on_baseline_point_clicked)
        baseline_layout.addWidget(self._baseline_viewer)
        
        splitter.addWidget(baseline_frame)
        
        # Follow-up panel
        followup_frame = QFrame()
        followup_layout = QVBoxLayout(followup_frame)
        followup_layout.setContentsMargins(0, 0, 0, 0)
        
        followup_label = QLabel("Follow-up Image")
        followup_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        followup_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        followup_layout.addWidget(followup_label)
        
        self._followup_viewer = PointSelectionViewer()
        self._followup_viewer.pointClicked.connect(self._on_followup_point_clicked)
        followup_layout.addWidget(self._followup_viewer)
        
        splitter.addWidget(followup_frame)
        
        layout.addWidget(splitter, 1)
        
        # Point status
        status_group = QGroupBox("Selected Points")
        status_layout = QHBoxLayout(status_group)
        
        self._point_labels = []
        for i in range(3):
            color = self.POINT_COLORS[i]
            label = QLabel(f"Point {i+1}: Not set")
            label.setStyleSheet(f"color: rgb({color.red()}, {color.green()}, {color.blue()});")
            status_layout.addWidget(label)
            self._point_labels.append(label)
        
        layout.addWidget(status_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self._clear_btn = QPushButton("Clear All Points")
        self._clear_btn.clicked.connect(self._clear_points)
        button_layout.addWidget(self._clear_btn)
        
        button_layout.addStretch()
        
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self._cancel_btn)
        
        self._register_btn = QPushButton("Register")
        self._register_btn.setEnabled(False)
        self._register_btn.clicked.connect(self._perform_registration)
        self._register_btn.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
                padding: 8px 20px;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #6c7086;
            }
        """)
        button_layout.addWidget(self._register_btn)
        
        layout.addLayout(button_layout)
    
    def _load_images(self) -> None:
        """Load images into viewers."""
        self._baseline_viewer.set_numpy_array(self._baseline.as_uint8())
        self._followup_viewer.set_numpy_array(self._followup.as_uint8())
        self._baseline_viewer.fit_in_view()
        self._followup_viewer.fit_in_view()
    
    def _update_instructions(self) -> None:
        """Update instruction text based on current state."""
        if self._current_point_index >= 3:
            self._instructions.setText("✓ All points selected! Click 'Register' to apply.")
            self._instructions.setStyleSheet("""
                QLabel {
                    font-size: 14pt;
                    font-weight: bold;
                    padding: 10px;
                    background-color: #a6e3a1;
                    color: #1e1e2e;
                    border-radius: 5px;
                }
            """)
        else:
            point_num = self._current_point_index + 1
            image_name = "BASELINE" if self._selecting_baseline else "FOLLOW-UP"
            color = self.POINT_COLORS[self._current_point_index]
            self._instructions.setText(
                f"Click Point {point_num} on {image_name} image"
            )
            self._instructions.setStyleSheet(f"""
                QLabel {{
                    font-size: 14pt;
                    font-weight: bold;
                    padding: 10px;
                    background-color: rgb({color.red()}, {color.green()}, {color.blue()});
                    color: white;
                    border-radius: 5px;
                }}
            """)
    
    def _on_baseline_point_clicked(self, point: QPointF) -> None:
        """Handle point click on baseline image."""
        if self._current_point_index >= 3:
            return
        
        if not self._selecting_baseline:
            return
        
        color = self.POINT_COLORS[self._current_point_index]
        
        # Add or update point pair
        if self._current_point_index >= len(self._point_pairs):
            self._point_pairs.append(PointPair(color=color))
        
        self._point_pairs[self._current_point_index].baseline_point = point
        
        # Add visual marker
        self._baseline_viewer.add_point(point, color, str(self._current_point_index + 1))
        
        # Switch to follow-up
        self._selecting_baseline = False
        self._update_instructions()
        self._update_point_labels()
    
    def _on_followup_point_clicked(self, point: QPointF) -> None:
        """Handle point click on follow-up image."""
        if self._current_point_index >= 3:
            return
        
        if self._selecting_baseline:
            return
        
        color = self.POINT_COLORS[self._current_point_index]
        
        # Update point pair
        self._point_pairs[self._current_point_index].followup_point = point
        
        # Add visual marker
        self._followup_viewer.add_point(point, color, str(self._current_point_index + 1))
        
        # Move to next point
        self._current_point_index += 1
        self._selecting_baseline = True
        
        self._update_instructions()
        self._update_point_labels()
        
        # Enable register button if all points selected
        if self._current_point_index >= 3:
            self._register_btn.setEnabled(True)
    
    def _update_point_labels(self) -> None:
        """Update point status labels."""
        for i, label in enumerate(self._point_labels):
            color = self.POINT_COLORS[i]
            if i < len(self._point_pairs):
                pair = self._point_pairs[i]
                if pair.baseline_point and pair.followup_point:
                    bp = pair.baseline_point
                    fp = pair.followup_point
                    label.setText(
                        f"Point {i+1}: ({bp.x():.0f}, {bp.y():.0f}) → ({fp.x():.0f}, {fp.y():.0f})"
                    )
                elif pair.baseline_point:
                    bp = pair.baseline_point
                    label.setText(f"Point {i+1}: ({bp.x():.0f}, {bp.y():.0f}) → ?")
                else:
                    label.setText(f"Point {i+1}: Not set")
            else:
                label.setText(f"Point {i+1}: Not set")
    
    def _clear_points(self) -> None:
        """Clear all selected points."""
        self._point_pairs.clear()
        self._current_point_index = 0
        self._selecting_baseline = True
        
        self._baseline_viewer.clear_points()
        self._followup_viewer.clear_points()
        
        self._register_btn.setEnabled(False)
        self._update_instructions()
        self._update_point_labels()
    
    def _perform_registration(self) -> None:
        """Compute and apply the affine transformation."""
        if not CV2_AVAILABLE:
            QMessageBox.critical(self, "Error", "OpenCV is required for registration.")
            return
        
        if len(self._point_pairs) < 3:
            QMessageBox.warning(self, "Warning", "Please select 3 point pairs.")
            return
        
        # Extract point coordinates
        src_points = []  # Follow-up points
        dst_points = []  # Baseline points
        
        for pair in self._point_pairs:
            if pair.baseline_point and pair.followup_point:
                dst_points.append([pair.baseline_point.x(), pair.baseline_point.y()])
                src_points.append([pair.followup_point.x(), pair.followup_point.y()])
        
        if len(src_points) < 3:
            QMessageBox.warning(self, "Warning", "Need 3 complete point pairs.")
            return
        
        src_pts = np.float32(src_points)
        dst_pts = np.float32(dst_points)
        
        try:
            # Compute affine transform
            transform_matrix = cv2.getAffineTransform(src_pts, dst_pts)
            
            # Apply transform
            followup_img = self._followup.pixel_array
            h, w = self._baseline.height, self._baseline.width
            
            registered_img = cv2.warpAffine(
                followup_img, transform_matrix, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # Create result
            result = TransformResult(
                transform_matrix=transform_matrix,
                motion_model=MotionModel.AFFINE,
                coarse_transform=transform_matrix.copy(),
                registration_time_ms=0,
                quality_metrics={"method": "manual_3point"},
                warnings=[]
            )
            
            # Create registered ImageData
            registered = self._followup.copy()
            registered.pixel_array = registered_img
            registered.preprocessing_history.append("manual_3point_registration")
            
            logger.info("Manual 3-point registration complete")
            
            self.registrationComplete.emit(result, registered)
            self.accept()
            
        except Exception as e:
            logger.exception("Manual registration failed")
            QMessageBox.critical(
                self, "Registration Error",
                f"Failed to compute transformation:\n{str(e)}"
            )
