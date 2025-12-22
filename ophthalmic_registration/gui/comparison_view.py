"""
Comparison view widget for registered image analysis.

Provides multiple comparison modes including overlay, difference,
checkerboard, and side-by-side views.
"""

from enum import Enum
from typing import Optional
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QLabel, QSlider, QComboBox, QPushButton, QFrame,
    QSizePolicy, QSplitter, QButtonGroup, QToolButton, QDockWidget
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal, QPointF

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ophthalmic_registration.core.image_data import ImageData
from ophthalmic_registration.gui.image_viewer import ImageViewer, ImagePanel


class ComparisonMode(Enum):
    """Available comparison modes."""
    SIDE_BY_SIDE = "Side by Side"
    OVERLAY = "Overlay"
    DIFFERENCE = "Difference"
    CHECKERBOARD = "Checkerboard"
    SPLIT = "Split View"


class ComparisonView(QWidget):
    """
    Multi-mode comparison view for baseline and registered images.
    
    Supports synchronized viewing across multiple comparison modes
    with interactive controls.
    
    Signals:
        modeChanged: Emitted when comparison mode changes
    """
    
    modeChanged = pyqtSignal(str)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._baseline: Optional[ImageData] = None
        self._registered: Optional[ImageData] = None
        self._current_mode = ComparisonMode.SIDE_BY_SIDE
        self._overlay_alpha = 0.5
        self._checker_size = 50
        self._use_native_colors = False
        self._show_registration_points = False
        self._registration_result = None
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the comparison view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Stacked widget for different views
        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)
        
        # Create view pages
        self._create_side_by_side_view()
        self._create_overlay_view()
        self._create_difference_view()
        self._create_checkerboard_view()
        self._create_split_view()
    
    def _create_side_by_side_view(self) -> None:
        """Create side-by-side comparison view."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self._sbs_baseline = ImagePanel("Baseline")
        splitter.addWidget(self._sbs_baseline)
        
        self._sbs_registered = ImagePanel("Registered")
        splitter.addWidget(self._sbs_registered)
        
        layout.addWidget(splitter)
        self._stack.addWidget(widget)
    
    def _create_overlay_view(self) -> None:
        """Create overlay comparison view."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._overlay_viewer = ImageViewer()
        layout.addWidget(self._overlay_viewer)
        
        self._stack.addWidget(widget)
    
    def _create_difference_view(self) -> None:
        """Create difference map view."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._diff_viewer = ImageViewer()
        layout.addWidget(self._diff_viewer)
        
        self._stack.addWidget(widget)
    
    def _create_checkerboard_view(self) -> None:
        """Create checkerboard comparison view."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._checker_viewer = ImageViewer()
        layout.addWidget(self._checker_viewer)
        
        self._stack.addWidget(widget)
    
    def _create_split_view(self) -> None:
        """Create split/curtain comparison view."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._split_viewer = ImageViewer()
        layout.addWidget(self._split_viewer)
        
        # Add slider for split position
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Baseline"))
        self._split_slider = QSlider(Qt.Orientation.Horizontal)
        self._split_slider.setRange(0, 100)
        self._split_slider.setValue(50)
        self._split_slider.valueChanged.connect(self._on_split_position_changed)
        slider_layout.addWidget(self._split_slider)
        slider_layout.addWidget(QLabel("Registered"))
        layout.addLayout(slider_layout)
        
        self._split_position = 0.5  # Default to middle
        
        self._stack.addWidget(widget)
    
    def set_images(
        self,
        baseline: ImageData,
        registered: ImageData
    ) -> None:
        """
        Set the images to compare.
        
        Args:
            baseline: Baseline/reference image
            registered: Registered follow-up image
        """
        self._baseline = baseline
        self._registered = registered
        
        # Update all views
        self._update_side_by_side()
        self._update_overlay()
        self._update_difference()
        self._update_checkerboard()
        self._update_split()
        
        # Fit to view after loading
        self.fit_to_view()
    
    def _update_side_by_side(self) -> None:
        """Update side-by-side view."""
        if self._baseline:
            self._sbs_baseline.set_image(self._baseline)
        if self._registered:
            self._sbs_registered.set_image(self._registered)
    
    def _update_overlay(self) -> None:
        """Update overlay view."""
        if not self._baseline or not self._registered:
            return
        
        if not CV2_AVAILABLE:
            return
        
        img1 = self._baseline.as_uint8()
        img2 = self._registered.as_uint8()
        
        # Ensure same size
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
        
        if self._use_native_colors:
            # Use native colors - simple alpha blending
            if img1.ndim == 2:
                img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
            else:
                img1_color = img1.copy()
            
            if img2.ndim == 2:
                img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
            else:
                img2_color = img2.copy()
        else:
            # Create colorized overlay (magenta/green)
            if img1.ndim == 2:
                img1_color = np.stack([img1, np.zeros_like(img1), img1], axis=-1)
            else:
                img1_color = img1.copy()
            
            if img2.ndim == 2:
                img2_color = np.stack([np.zeros_like(img2), img2, np.zeros_like(img2)], axis=-1)
            else:
                img2_color = img2.copy()
        
        alpha = self._overlay_alpha
        overlay = cv2.addWeighted(
            img1_color.astype(np.float32), 1 - alpha,
            img2_color.astype(np.float32), alpha,
            0
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        self._overlay_viewer.set_numpy_array(overlay)
        # Preserve pixel spacing from baseline
        if self._baseline.metadata and self._baseline.metadata.pixel_spacing:
            self._overlay_viewer.set_pixel_spacing(self._baseline.metadata.pixel_spacing)
    
    def _update_difference(self) -> None:
        """Update difference view."""
        if not self._baseline or not self._registered:
            return
        
        if not CV2_AVAILABLE:
            return
        
        img1 = self._baseline.as_uint8()
        img2 = self._registered.as_uint8()
        
        # Ensure same size
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
        
        # Convert to grayscale if needed
        if img1.ndim == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if img2.ndim == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Compute absolute difference
        diff = cv2.absdiff(img1, img2)
        
        # Apply colormap for visualization
        diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)
        
        self._diff_viewer.set_numpy_array(diff_colored)
        # Preserve pixel spacing from baseline
        if self._baseline.metadata and self._baseline.metadata.pixel_spacing:
            self._diff_viewer.set_pixel_spacing(self._baseline.metadata.pixel_spacing)
    
    def _update_checkerboard(self) -> None:
        """Update checkerboard view."""
        if not self._baseline or not self._registered:
            return
        
        img1 = self._baseline.as_uint8()
        img2 = self._registered.as_uint8()
        
        # Ensure same size
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
        
        # Create checkerboard mask
        grid_size = self._checker_size
        mask = np.zeros((h, w), dtype=bool)
        for i in range(0, h, grid_size):
            for j in range(0, w, grid_size):
                if ((i // grid_size) + (j // grid_size)) % 2 == 0:
                    mask[i:i+grid_size, j:j+grid_size] = True
        
        # Apply mask
        if img1.ndim == 2:
            result = np.where(mask, img1, img2)
        else:
            result = np.where(mask[:, :, np.newaxis], img1, img2)
        
        self._checker_viewer.set_numpy_array(result.astype(np.uint8))
        # Preserve pixel spacing from baseline
        if self._baseline.metadata and self._baseline.metadata.pixel_spacing:
            self._checker_viewer.set_pixel_spacing(self._baseline.metadata.pixel_spacing)
    
    def _on_split_position_changed(self, value: int) -> None:
        """Handle split position slider change."""
        self._split_position = value / 100.0
        self._update_split()
    
    def _update_split(self) -> None:
        """Update split/curtain view."""
        if not self._baseline or not self._registered:
            return
        
        if not CV2_AVAILABLE:
            return
        
        img1 = self._baseline.as_uint8()
        img2 = self._registered.as_uint8()
        
        # Ensure same size
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
        
        # Convert RGBA to RGB if needed
        if img1.ndim == 3 and img1.shape[2] == 4:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGBA2RGB)
        if img2.ndim == 3 and img2.shape[2] == 4:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGBA2RGB)
        
        # Create split view using slider position
        split_pos = int(w * self._split_position)
        
        if img1.ndim == 2:
            result = np.zeros((h, w), dtype=np.uint8)
            result[:, :split_pos] = img1[:, :split_pos]
            result[:, split_pos:] = img2[:, split_pos:]
        else:
            channels = img1.shape[2] if img1.ndim == 3 else 3
            result = np.zeros((h, w, channels), dtype=np.uint8)
            result[:, :split_pos] = img1[:, :split_pos]
            result[:, split_pos:] = img2[:, split_pos:]
        
        # Draw split line
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        cv2.line(result, (split_pos, 0), (split_pos, h), (255, 255, 0), 2)
        
        self._split_viewer.set_numpy_array(result)
        # Preserve pixel spacing from baseline
        if self._baseline.metadata and self._baseline.metadata.pixel_spacing:
            self._split_viewer.set_pixel_spacing(self._baseline.metadata.pixel_spacing)
    
    def set_mode(self, mode_name: str) -> None:
        """
        Set the comparison mode by name.
        
        Args:
            mode_name: Mode name string (e.g., "Side by Side", "Overlay")
        """
        # Map mode name to enum
        mode_map = {
            "Side by Side": ComparisonMode.SIDE_BY_SIDE,
            "Overlay": ComparisonMode.OVERLAY,
            "Difference": ComparisonMode.DIFFERENCE,
            "Checkerboard": ComparisonMode.CHECKERBOARD,
            "Split View": ComparisonMode.SPLIT,
        }
        
        mode = mode_map.get(mode_name)
        if mode is None:
            return
        
        self._current_mode = mode
        
        # Update stack
        mode_to_index = {
            ComparisonMode.SIDE_BY_SIDE: 0,
            ComparisonMode.OVERLAY: 1,
            ComparisonMode.DIFFERENCE: 2,
            ComparisonMode.CHECKERBOARD: 3,
            ComparisonMode.SPLIT: 4,
        }
        self._stack.setCurrentIndex(mode_to_index[self._current_mode])
        self.modeChanged.emit(self._current_mode.value)
        
        # Fit to view when mode changes
        self.fit_to_view()
    
    def set_overlay_alpha(self, value: int) -> None:
        """
        Set overlay alpha value (0-100).
        
        Args:
            value: Alpha value from 0 to 100
        """
        self._overlay_alpha = value / 100.0
        self._update_overlay()
    
    def set_native_colors(self, use_native: bool) -> None:
        """
        Set whether to use native colors in overlay mode.
        
        Args:
            use_native: True to use native colors, False for colored overlay
        """
        self._use_native_colors = use_native
        self._update_overlay()
    
    def set_checker_size(self, size: int) -> None:
        """
        Set checkerboard grid size.
        
        Args:
            size: Grid size in pixels
        """
        self._checker_size = size
        self._update_checkerboard()
    
    def clear(self) -> None:
        """Clear all views."""
        self._baseline = None
        self._registered = None
        
        self._sbs_baseline.clear()
        self._sbs_registered.clear()
        self._overlay_viewer.clear()
        self._diff_viewer.clear()
        self._checker_viewer.clear()
        self._split_viewer.clear()
    
    def fit_to_view(self) -> None:
        """Fit current view to screen."""
        if self._current_mode == ComparisonMode.SIDE_BY_SIDE:
            self._sbs_baseline.fit_to_view()
            self._sbs_registered.fit_to_view()
        elif self._current_mode == ComparisonMode.OVERLAY:
            self._overlay_viewer.fit_in_view()
        elif self._current_mode == ComparisonMode.DIFFERENCE:
            self._diff_viewer.fit_in_view()
        elif self._current_mode == ComparisonMode.CHECKERBOARD:
            self._checker_viewer.fit_in_view()
        elif self._current_mode == ComparisonMode.SPLIT:
            self._split_viewer.fit_in_view()
    
    def zoom_100(self) -> None:
        """Set zoom to 100% (1:1) for current view."""
        if self._current_mode == ComparisonMode.SIDE_BY_SIDE:
            self._sbs_baseline.zoom_100()
            self._sbs_registered.zoom_100()
        elif self._current_mode == ComparisonMode.OVERLAY:
            self._overlay_viewer.zoom_100()
        elif self._current_mode == ComparisonMode.DIFFERENCE:
            self._diff_viewer.zoom_100()
        elif self._current_mode == ComparisonMode.CHECKERBOARD:
            self._checker_viewer.zoom_100()
        elif self._current_mode == ComparisonMode.SPLIT:
            self._split_viewer.zoom_100()
    
    def set_registration_result(self, result) -> None:
        """Store registration result for point visualization."""
        self._registration_result = result
        if self._show_registration_points:
            self._update_registration_points()
    
    def set_show_registration_points(self, show: bool) -> None:
        """Toggle registration points visualization."""
        self._show_registration_points = show
        if show:
            self._update_registration_points()
        else:
            self._clear_registration_points()
    
    def _update_registration_points(self) -> None:
        """Draw registration points on side by side view."""
        if not self._registration_result or not self._registration_result.feature_match_result:
            return
        
        # Only show in side by side view
        if self._current_mode != ComparisonMode.SIDE_BY_SIDE:
            return
        
        if not self._baseline or not self._registered:
            return
        
        feature_result = self._registration_result.feature_match_result
        
        # Get inlier matches
        if feature_result.inlier_mask is None or len(feature_result.matches) == 0:
            return
        
        if not CV2_AVAILABLE:
            return
        
        # Draw points on baseline image
        img1 = self._baseline.as_uint8()
        if img1.ndim == 2:
            img1_with_points = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        else:
            img1_with_points = img1.copy()
        
        # Draw points on registered image
        img2 = self._registered.as_uint8()
        if img2.ndim == 2:
            img2_with_points = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        else:
            img2_with_points = img2.copy()
        
        # Get inlier matches with their coordinates
        inlier_matches = []
        for i, match in enumerate(feature_result.matches):
            if feature_result.inlier_mask[i]:
                pt = feature_result.keypoints_baseline[match.queryIdx].pt
                inlier_matches.append((i, match, match.distance, pt[0], pt[1]))
        
        # Select spatially distributed points across the image
        # Divide image into a 5x5 grid and randomly select points from each cell
        import random
        h, w = img1_with_points.shape[:2]
        grid_rows, grid_cols = 5, 5
        cell_h, cell_w = h / grid_rows, w / grid_cols
        
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
        
        # Draw registration points on each image separately (no combined image)
        for idx, (i, match, dist) in enumerate(top_matches):
            # Get keypoint coordinates
            pt1 = feature_result.keypoints_baseline[match.queryIdx].pt
            pt2 = feature_result.keypoints_followup[match.trainIdx].pt
            
            # Convert to integer coordinates
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])
            
            # Draw circles at keypoints with number labels
            cv2.circle(img1_with_points, (x1, y1), 5, (0, 255, 255), -1)  # Cyan
            cv2.circle(img1_with_points, (x1, y1), 7, (255, 255, 0), 2)  # Yellow outline
            cv2.putText(img1_with_points, str(idx + 1), (x1 + 8, y1 + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.circle(img2_with_points, (x2, y2), 5, (0, 255, 255), -1)  # Cyan
            cv2.circle(img2_with_points, (x2, y2), 7, (255, 255, 0), 2)  # Yellow outline
            cv2.putText(img2_with_points, str(idx + 1), (x2 + 8, y2 + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Ensure arrays are contiguous for QImage
        img1_with_points = np.ascontiguousarray(img1_with_points)
        img2_with_points = np.ascontiguousarray(img2_with_points)
        
        # Update the side-by-side viewers with annotated images
        self._sbs_baseline.viewer.set_numpy_array(img1_with_points)
        if self._baseline.metadata and self._baseline.metadata.pixel_spacing:
            self._sbs_baseline.viewer.set_pixel_spacing(self._baseline.metadata.pixel_spacing)
        
        self._sbs_registered.viewer.set_numpy_array(img2_with_points)
        if self._registered.metadata and self._registered.metadata.pixel_spacing:
            self._sbs_registered.viewer.set_pixel_spacing(self._registered.metadata.pixel_spacing)
    
    def _clear_registration_points(self) -> None:
        """Clear registration points visualization."""
        # Redraw the side by side view without points
        if self._current_mode == ComparisonMode.SIDE_BY_SIDE:
            self._update_side_by_side()
    
    def get_current_mode(self) -> str:
        """Get the current comparison mode name."""
        return self._current_mode.value
