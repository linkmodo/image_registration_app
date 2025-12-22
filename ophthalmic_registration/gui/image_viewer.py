"""
Image viewer widget with zoom, pan, and overlay capabilities.

Provides a high-quality image display component suitable for
medical imaging applications.
"""

from typing import Optional, Tuple, List
from enum import Enum
import numpy as np
import math

from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QSizePolicy, QMenu, QFileDialog, QGraphicsLineItem,
    QGraphicsEllipseItem, QGraphicsPolygonItem, QGraphicsTextItem
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent,
    QColor, QPen, QBrush, QAction, QCursor, QPolygonF, QFont
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF

from ophthalmic_registration.core.image_data import ImageData, PixelSpacing
from ophthalmic_registration.gui.measurement_items_new import (
    DistanceMeasurement, AreaMeasurement
)


class MeasurementMode(Enum):
    """Measurement modes for the viewer."""
    NONE = "none"
    DISTANCE = "distance"
    AREA = "area"


class ImageViewer(QGraphicsView):
    """
    Advanced image viewer with zoom, pan, and measurement capabilities.
    
    Signals:
        imageLoaded: Emitted when an image is loaded
        positionChanged: Emitted when mouse position changes (x, y, value)
        zoomChanged: Emitted when zoom level changes
        measurementRequested: Emitted when user requests a measurement
    """
    
    imageLoaded = pyqtSignal(object)  # ImageData
    positionChanged = pyqtSignal(int, int, float)  # x, y, pixel_value
    zoomChanged = pyqtSignal(float)
    measurementRequested = pyqtSignal(list)  # List of points
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._image_data: Optional[ImageData] = None
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 20.0
        
        # Interaction state
        self._panning = False
        self._last_pan_pos = QPointF()
        self._measuring = False
        self._measure_points: List[QPointF] = []
        
        # Measurement state
        self._measurement_mode = MeasurementMode.NONE
        self._measurement_items: List = []  # Graphics items for measurements
        self._interactive_measurements: List = []  # Interactive measurement objects
        self._current_measurement_points: List[QPointF] = []
        self._pixel_spacing: Optional[PixelSpacing] = None  # For mm calculations
        self._preview_line: Optional[QGraphicsLineItem] = None  # Preview line for distance
        self._preview_polygon: Optional[QGraphicsPolygonItem] = None  # Preview polygon for area
        self._area_drawing = False  # Track if user is drawing area
        
        # Configure view
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor("#11111b")))
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        # Enable mouse tracking for position reporting
        self.setMouseTracking(True)
        
        # Context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
    
    def set_image(self, image_data: ImageData) -> None:
        """
        Set the image to display.
        
        Args:
            image_data: ImageData object to display
        """
        self._image_data = image_data
        # Extract pixel spacing from metadata if available
        if image_data.metadata and image_data.metadata.pixel_spacing:
            self._pixel_spacing = image_data.metadata.pixel_spacing
        self._update_pixmap()
        self.imageLoaded.emit(image_data)
    
    def set_pixel_spacing(self, pixel_spacing: Optional[PixelSpacing]) -> None:
        """Set pixel spacing for accurate measurements."""
        self._pixel_spacing = pixel_spacing
    
    def set_numpy_array(self, array: np.ndarray) -> None:
        """Set image from numpy array directly."""
        self._image_data = ImageData(pixel_array=array)
        # Don't reset pixel spacing when setting numpy array
        self._update_pixmap()
    
    def _update_pixmap(self) -> None:
        """Update the displayed pixmap from current image data."""
        if self._image_data is None:
            return
        
        # Convert to uint8 for display
        img_array = self._image_data.as_uint8()
        
        # Create QImage
        if img_array.ndim == 2:
            # Grayscale
            h, w = img_array.shape
            bytes_per_line = w
            qimage = QImage(
                img_array.data, w, h, bytes_per_line,
                QImage.Format.Format_Grayscale8
            )
        elif img_array.ndim == 3 and img_array.shape[2] == 3:
            # RGB
            h, w, c = img_array.shape
            bytes_per_line = 3 * w
            qimage = QImage(
                img_array.data, w, h, bytes_per_line,
                QImage.Format.Format_RGB888
            )
        elif img_array.ndim == 3 and img_array.shape[2] == 4:
            # RGBA
            h, w, c = img_array.shape
            bytes_per_line = 4 * w
            qimage = QImage(
                img_array.data, w, h, bytes_per_line,
                QImage.Format.Format_RGBA8888
            )
        else:
            return
        
        pixmap = QPixmap.fromImage(qimage)
        
        # Update scene - preserve measurement items
        if self._pixmap_item:
            self._scene.removeItem(self._pixmap_item)
        else:
            self._scene.clear()
            self._measurement_items = []
        
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(-10)  # Keep pixmap behind measurements
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        
        # Fit in view initially
        self.fit_in_view()
    
    def fit_in_view(self) -> None:
        """Fit the image to the view while maintaining aspect ratio."""
        if self._pixmap_item is None:
            return
        
        self.fitInView(
            self._scene.sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self._update_zoom_factor()
    
    def zoom_in(self) -> None:
        """Zoom in by 25%."""
        self._apply_zoom(1.25)
    
    def zoom_out(self) -> None:
        """Zoom out by 25%."""
        self._apply_zoom(0.8)
    
    def zoom_100(self) -> None:
        """Reset to 100% zoom."""
        if self._pixmap_item is None:
            return
        
        self.resetTransform()
        self._zoom_factor = 1.0
        self.zoomChanged.emit(self._zoom_factor)
    
    def _apply_zoom(self, factor: float) -> None:
        """Apply zoom factor with bounds checking."""
        new_zoom = self._zoom_factor * factor
        
        if self._min_zoom <= new_zoom <= self._max_zoom:
            self.scale(factor, factor)
            self._zoom_factor = new_zoom
            self.zoomChanged.emit(self._zoom_factor)
    
    def _update_zoom_factor(self) -> None:
        """Update internal zoom factor from current transform."""
        self._zoom_factor = self.transform().m11()
        self.zoomChanged.emit(self._zoom_factor)
    
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming."""
        if event.angleDelta().y() > 0:
            self._apply_zoom(1.15)
        else:
            self._apply_zoom(0.87)
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press for panning and measuring."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._last_pan_pos = event.position()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        elif event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on an existing measurement item first
            item = self.itemAt(event.pos())
            if item and self._is_measurement_item(item):
                # Let the item handle the event (for dragging/selecting)
                super().mousePressEvent(event)
                return
            
            # Handle new measurements only if not clicking on existing ones
            if self._measurement_mode == MeasurementMode.DISTANCE:
                scene_pos = self.mapToScene(event.pos())
                self._handle_distance_click(scene_pos)
            elif self._measurement_mode == MeasurementMode.AREA:
                scene_pos = self.mapToScene(event.pos())
                self._handle_area_press(scene_pos)
            elif self._measuring:
                scene_pos = self.mapToScene(event.pos())
                self._measure_points.append(scene_pos)
                if len(self._measure_points) == 2:
                    self.measurementRequested.emit(self._measure_points)
                    self._measure_points = []
            else:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)
    
    def _is_measurement_item(self, item) -> bool:
        """Check if an item is part of a measurement."""
        # Check if item belongs to any measurement
        for measurement in self._interactive_measurements:
            if hasattr(measurement, 'contains_item') and measurement.contains_item(item):
                return True
        return False
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse movement for panning, position tracking, and measurement preview."""
        if self._panning:
            delta = event.position() - self._last_pan_pos
            self._last_pan_pos = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
        elif self._measurement_mode == MeasurementMode.DISTANCE and len(self._current_measurement_points) == 1:
            # Show distance preview line
            scene_pos = self.mapToScene(event.pos())
            self._update_distance_preview(scene_pos)
        elif self._measurement_mode == MeasurementMode.AREA and len(self._current_measurement_points) > 0:
            # Show area preview while dragging
            if event.buttons() & Qt.MouseButton.LeftButton:
                scene_pos = self.mapToScene(event.pos())
                self._update_area_drag(scene_pos)
        elif self._image_data is not None:
            # Emit position and pixel value
            scene_pos = self.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())
            
            if (0 <= x < self._image_data.width and 
                0 <= y < self._image_data.height):
                value = self._image_data.pixel_array[y, x]
                if isinstance(value, np.ndarray):
                    value = value[0]
                self.positionChanged.emit(x, y, float(value))
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        elif event.button() == Qt.MouseButton.LeftButton and self._measurement_mode == MeasurementMode.AREA:
            if self._area_drawing:
                self._complete_area_measurement()
        super().mouseReleaseEvent(event)
    
    def _show_context_menu(self, pos) -> None:
        """Show context menu."""
        menu = QMenu(self)
        
        fit_action = QAction("Fit to View", self)
        fit_action.triggered.connect(self.fit_in_view)
        menu.addAction(fit_action)
        
        zoom_100_action = QAction("Zoom 100%", self)
        zoom_100_action.triggered.connect(self.zoom_100)
        menu.addAction(zoom_100_action)
        
        menu.addSeparator()
        
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        menu.addAction(zoom_out_action)
        
        # Measurement tools
        if self._image_data is not None:
            menu.addSeparator()
            
            measure_menu = menu.addMenu("Measure")
            
            distance_action = QAction("📏 Distance", self)
            distance_action.setCheckable(True)
            distance_action.setChecked(self._measurement_mode == MeasurementMode.DISTANCE)
            distance_action.triggered.connect(lambda: self._set_measurement_mode(MeasurementMode.DISTANCE))
            measure_menu.addAction(distance_action)
            
            area_action = QAction("⬜ Area", self)
            area_action.setCheckable(True)
            area_action.setChecked(self._measurement_mode == MeasurementMode.AREA)
            area_action.triggered.connect(lambda: self._set_measurement_mode(MeasurementMode.AREA))
            measure_menu.addAction(area_action)
            
            measure_menu.addSeparator()
            
            clear_action = QAction("Clear Measurements", self)
            clear_action.triggered.connect(self.clear_measurements)
            measure_menu.addAction(clear_action)
        
        menu.exec(self.mapToGlobal(pos))
    
    def _set_measurement_mode(self, mode: MeasurementMode) -> None:
        """Set the measurement mode."""
        if self._measurement_mode == mode:
            # Toggle off if same mode
            self._measurement_mode = MeasurementMode.NONE
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        else:
            self._measurement_mode = mode
            self._current_measurement_points = []
            if mode == MeasurementMode.DISTANCE:
                self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            elif mode == MeasurementMode.AREA:
                self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
    
    def _handle_distance_click(self, scene_pos: QPointF) -> None:
        """Handle distance measurement click."""
        if len(self._current_measurement_points) == 0:
            # First click - record point (no preview dot needed, measurement will have its own)
            self._current_measurement_points.append(scene_pos)
        elif len(self._current_measurement_points) == 1:
            # Second click - complete measurement
            self._current_measurement_points.append(scene_pos)
            # Remove preview line
            if self._preview_line:
                self._scene.removeItem(self._preview_line)
                self._preview_line = None
            self._complete_distance_measurement()
    
    def _handle_area_press(self, scene_pos: QPointF) -> None:
        """Handle area measurement mouse press - start drawing."""
        self._area_drawing = True
        self._current_measurement_points = [scene_pos]
    
    def _update_distance_preview(self, current_pos: QPointF) -> None:
        """Update preview line for distance measurement."""
        if len(self._current_measurement_points) != 1:
            return
        
        # Remove old preview
        if self._preview_line:
            self._scene.removeItem(self._preview_line)
        
        # Draw new preview line
        p1 = self._current_measurement_points[0]
        self._preview_line = self._scene.addLine(
            p1.x(), p1.y(), current_pos.x(), current_pos.y(),
            QPen(QColor("#a6e3a1"), 1, Qt.PenStyle.DashLine)
        )
        self._preview_line.setZValue(10)
    
    def _update_area_drag(self, current_pos: QPointF) -> None:
        """Update area polygon while dragging."""
        if not self._area_drawing or len(self._current_measurement_points) == 0:
            return
        
        # Add point if it's far enough from last point (no dots, just track points)
        if len(self._current_measurement_points) > 0:
            last_pt = self._current_measurement_points[-1]
            dist = math.sqrt((current_pos.x() - last_pt.x())**2 + (current_pos.y() - last_pt.y())**2)
            if dist > 5:  # Minimum distance threshold
                self._current_measurement_points.append(current_pos)
        
        # Update preview polygon (lines only, no dots)
        if len(self._current_measurement_points) >= 2:
            # Remove old preview
            if self._preview_polygon:
                self._scene.removeItem(self._preview_polygon)
            
            # Create polygon with closing line
            polygon = QPolygonF()
            for pt in self._current_measurement_points:
                polygon.append(pt)
            # Close the polygon
            polygon.append(self._current_measurement_points[0])
            
            self._preview_polygon = self._scene.addPolygon(
                polygon,
                QPen(QColor("#89b4fa"), 1, Qt.PenStyle.DashLine),
                QBrush(QColor(137, 180, 250, 30))
            )
            self._preview_polygon.setZValue(5)
    
    def _complete_distance_measurement(self) -> None:
        """Complete a distance measurement."""
        if len(self._current_measurement_points) < 2:
            return
        
        p1 = self._current_measurement_points[0]
        p2 = self._current_measurement_points[1]
        
        # Create interactive distance measurement
        measurement = DistanceMeasurement(self._scene, p1, p2, self._pixel_spacing)
        measurement.signals.deleteRequested.connect(self._delete_measurement)
        self._interactive_measurements.append(measurement)
        
        # Reset for next measurement
        self._current_measurement_points = []
    
    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Handle double-click - not used for area measurement anymore."""
        super().mouseDoubleClickEvent(event)
    
    def _complete_area_measurement(self) -> None:
        """Complete an area measurement when mouse is released."""
        self._area_drawing = False
        
        if len(self._current_measurement_points) < 3:
            # Not enough points, clear and return
            self._current_measurement_points = []
            if self._preview_polygon:
                self._scene.removeItem(self._preview_polygon)
                self._preview_polygon = None
            return
        
        # Remove preview
        if self._preview_polygon:
            self._scene.removeItem(self._preview_polygon)
            self._preview_polygon = None
        
        # Create interactive area measurement
        measurement = AreaMeasurement(self._scene, self._current_measurement_points, self._pixel_spacing)
        measurement.signals.deleteRequested.connect(self._delete_measurement)
        self._interactive_measurements.append(measurement)
        
        # Reset
        self._current_measurement_points = []
    
    def _delete_measurement(self, measurement):
        """Delete a measurement from the scene."""
        if measurement in self._interactive_measurements:
            self._interactive_measurements.remove(measurement)
        measurement.remove_from_scene()
    
    def clear_measurements(self) -> None:
        """Clear all measurement annotations."""
        for item in self._measurement_items:
            self._scene.removeItem(item)
        self._measurement_items = []
        
        # Clear interactive measurements
        for measurement in self._interactive_measurements:
            measurement.remove_from_scene()
        self._interactive_measurements = []
        
        self._current_measurement_points = []
        if self._preview_line:
            self._scene.removeItem(self._preview_line)
            self._preview_line = None
        if self._preview_polygon:
            self._scene.removeItem(self._preview_polygon)
            self._preview_polygon = None
        self._measurement_mode = MeasurementMode.NONE
        self._area_drawing = False
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
    
    @property
    def image_data(self) -> Optional[ImageData]:
        """Get the current image data."""
        return self._image_data
    
    @property
    def zoom_factor(self) -> float:
        """Get the current zoom factor."""
        return self._zoom_factor
    
    def set_pixel_spacing(self, pixel_spacing: PixelSpacing) -> None:
        """Set pixel spacing for measurement calculations."""
        self._pixel_spacing = pixel_spacing
    
    def clear(self) -> None:
        """Clear the viewer."""
        self._scene.clear()
        self._pixmap_item = None
        self._image_data = None
        self._pixel_spacing = None
        self._zoom_factor = 1.0


class ImagePanel(QFrame):
    """
    Complete image panel with viewer, title, and info bar.
    
    Provides a self-contained panel for displaying an image with
    metadata and interaction feedback.
    """
    
    imageDropped = pyqtSignal(str)  # File path
    browseClicked = pyqtSignal()  # Click to browse
    
    # Supported file filters
    FILE_FILTER = (
        "All Supported Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.dcm *.dicom);;"
        "DICOM Files (*.dcm *.dicom);;"
        "PNG Files (*.png);;"
        "JPEG Files (*.jpg *.jpeg);;"
        "TIFF Files (*.tiff *.tif);;"
        "BMP Files (*.bmp);;"
        "All Files (*)"
    )
    
    def __init__(
        self,
        title: str = "Image",
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self._title = title
        self._image_data: Optional[ImageData] = None
        
        self._setup_ui()
        self.setAcceptDrops(True)
    
    def _setup_ui(self) -> None:
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Title bar
        title_bar = QWidget()
        title_bar.setStyleSheet("background-color: #181825; padding: 8px;")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(12, 8, 12, 8)
        
        self._title_label = QLabel(self._title)
        self._title_label.setObjectName("titleLabel")
        title_layout.addWidget(self._title_label)
        
        title_layout.addStretch()
        
        self._info_label = QLabel("No image loaded")
        self._info_label.setObjectName("subtitleLabel")
        title_layout.addWidget(self._info_label)
        
        layout.addWidget(title_bar)
        
        # Image viewer
        self._viewer = ImageViewer()
        self._viewer.positionChanged.connect(self._on_position_changed)
        self._viewer.zoomChanged.connect(self._on_zoom_changed)
        layout.addWidget(self._viewer, 1)
        
        # Status bar
        status_bar = QWidget()
        status_bar.setStyleSheet("background-color: #181825; padding: 4px;")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(12, 4, 12, 4)
        
        self._position_label = QLabel("X: -- Y: -- Value: --")
        self._position_label.setObjectName("subtitleLabel")
        status_layout.addWidget(self._position_label)
        
        status_layout.addStretch()
        
        self._zoom_label = QLabel("Zoom: 100%")
        self._zoom_label.setObjectName("subtitleLabel")
        status_layout.addWidget(self._zoom_label)
        
        layout.addWidget(status_bar)
        
        # Initial state - show drop hint
        self.setObjectName("imagePanel")
        
        # Drop overlay
        self._drop_hint = QLabel("Drop image here\nor click to browse")
        self._drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_hint.setStyleSheet("""
            color: #6c7086;
            font-size: 12pt;
            background-color: transparent;
        """)
        self._viewer.setVisible(False)
        layout.addWidget(self._drop_hint, 1)
    
    def set_image(self, image_data: ImageData) -> None:
        """Set the image to display."""
        self._image_data = image_data
        self._viewer.set_image(image_data)
        
        # Update UI
        self._drop_hint.setVisible(False)
        self._viewer.setVisible(True)
        self.setObjectName("imagePanelLoaded")
        self.style().unpolish(self)
        self.style().polish(self)
        
        # Update info
        h, w = image_data.height, image_data.width
        info = f"{w} × {h}"
        if image_data.pixel_spacing:
            # Handle both PixelSpacing object and dict
            ps = image_data.pixel_spacing
            if hasattr(ps, 'mean_spacing'):
                spacing = ps.mean_spacing
            elif isinstance(ps, dict):
                spacing = (ps.get('row_spacing', 0) + ps.get('column_spacing', 0)) / 2
            else:
                spacing = None
            if spacing:
                info += f" | {spacing:.4f} mm/px"
        self._info_label.setText(info)
    
    def clear(self) -> None:
        """Clear the panel."""
        self._image_data = None
        self._viewer.clear()
        self._viewer.setVisible(False)
        self._drop_hint.setVisible(True)
        self.setObjectName("imagePanel")
        self.style().unpolish(self)
        self.style().polish(self)
        self._info_label.setText("No image loaded")
    
    def _on_position_changed(self, x: int, y: int, value: float) -> None:
        """Update position label."""
        self._position_label.setText(f"X: {x} Y: {y} Value: {value:.1f}")
    
    def _on_zoom_changed(self, zoom: float) -> None:
        """Update zoom label."""
        self._zoom_label.setText(f"Zoom: {zoom*100:.0f}%")
    
    def dragEnterEvent(self, event) -> None:
        """Handle drag enter."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event) -> None:
        """Handle drop."""
        urls = event.mimeData().urls()
        if urls:
            filepath = urls[0].toLocalFile()
            self.imageDropped.emit(filepath)
    
    def mousePressEvent(self, event) -> None:
        """Handle mouse press - click to browse when no image loaded."""
        if self._image_data is None and event.button() == Qt.MouseButton.LeftButton:
            self._browse_for_file()
        super().mousePressEvent(event)
    
    def _browse_for_file(self) -> None:
        """Open file browser dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            f"Open {self._title}",
            "",
            self.FILE_FILTER
        )
        if filepath:
            self.imageDropped.emit(filepath)
    
    @property
    def viewer(self) -> ImageViewer:
        """Get the image viewer."""
        return self._viewer
    
    @property
    def image_data(self) -> Optional[ImageData]:
        """Get the current image data."""
        return self._image_data
    
    def set_title(self, title: str) -> None:
        """Set the panel title."""
        self._title = title
        self._title_label.setText(title)
    
    def fit_to_view(self) -> None:
        """Fit image to view."""
        self._viewer.fit_in_view()
    
    def zoom_100(self) -> None:
        """Set zoom to 100%."""
        self._viewer.zoom_100()
