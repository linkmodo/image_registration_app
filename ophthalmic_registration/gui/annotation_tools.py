"""
Annotation and measurement tools for registered images.

Provides interactive tools for measuring distances, areas, and
adding annotations to ophthalmic images.
"""

from enum import Enum
from typing import Optional, List, Tuple
from dataclasses import dataclass
import math
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolButton, QLabel,
    QButtonGroup, QFrame, QListWidget, QListWidgetItem,
    QPushButton, QGroupBox, QDoubleSpinBox, QComboBox,
    QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsPolygonItem,
    QGraphicsTextItem, QGraphicsPathItem, QGraphicsItem
)
from PyQt6.QtGui import (
    QPen, QBrush, QColor, QFont, QPainterPath, QPolygonF
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF

from ophthalmic_registration.core.image_data import PixelSpacing


class AnnotationTool(Enum):
    """Available annotation tools."""
    NONE = "none"
    DISTANCE = "distance"
    AREA = "area"
    ANGLE = "angle"
    POINT = "point"
    FREEHAND = "freehand"


@dataclass
class Annotation:
    """Represents a single annotation."""
    tool: AnnotationTool
    points: List[QPointF]
    pixel_value: float = 0.0
    real_value: float = 0.0
    unit: str = "px"
    label: str = ""
    color: QColor = None
    graphics_items: List[QGraphicsItem] = None
    
    def __post_init__(self):
        if self.color is None:
            self.color = QColor("#89b4fa")
        if self.graphics_items is None:
            self.graphics_items = []
    
    @property
    def display_text(self) -> str:
        """Get display text for the annotation."""
        if self.tool == AnnotationTool.DISTANCE:
            if self.unit == "px":
                return f"{self.pixel_value:.1f} px"
            else:
                return f"{self.real_value:.3f} {self.unit}"
        elif self.tool == AnnotationTool.AREA:
            if self.unit == "px":
                return f"{self.pixel_value:.1f} px²"
            else:
                return f"{self.real_value:.4f} {self.unit}²"
        elif self.tool == AnnotationTool.ANGLE:
            return f"{self.real_value:.1f}°"
        elif self.tool == AnnotationTool.POINT:
            return f"({self.points[0].x():.0f}, {self.points[0].y():.0f})"
        return self.label


class AnnotationManager:
    """
    Manages annotations on an image viewer.
    
    Handles creation, display, and measurement calculations
    for various annotation types.
    """
    
    # Colors for different annotation types
    COLORS = {
        AnnotationTool.DISTANCE: QColor("#89b4fa"),  # Blue
        AnnotationTool.AREA: QColor("#a6e3a1"),      # Green
        AnnotationTool.ANGLE: QColor("#f9e2af"),     # Yellow
        AnnotationTool.POINT: QColor("#f38ba8"),    # Red
        AnnotationTool.FREEHAND: QColor("#cba6f7"), # Purple
    }
    
    def __init__(self, scene, pixel_spacing: Optional[PixelSpacing] = None):
        """
        Initialize annotation manager.
        
        Args:
            scene: QGraphicsScene to draw on
            pixel_spacing: Optional pixel spacing for real-world measurements
        """
        self._scene = scene
        self._pixel_spacing = pixel_spacing
        self._annotations: List[Annotation] = []
        self._current_tool = AnnotationTool.NONE
        self._current_points: List[QPointF] = []
        self._temp_items: List[QGraphicsItem] = []
        self._pen_width = 2
        self._font_size = 12
    
    def set_pixel_spacing(self, spacing: Optional[PixelSpacing]) -> None:
        """Set pixel spacing for measurements."""
        self._pixel_spacing = spacing
        # Recalculate all annotations
        for ann in self._annotations:
            self._calculate_measurement(ann)
            self._update_annotation_label(ann)
    
    def set_tool(self, tool: AnnotationTool) -> None:
        """Set the current annotation tool."""
        self._current_tool = tool
        self._clear_temp_items()
        self._current_points = []
    
    def add_point(self, point: QPointF) -> Optional[Annotation]:
        """
        Add a point to the current annotation.
        
        Returns completed annotation if finished, None otherwise.
        """
        if self._current_tool == AnnotationTool.NONE:
            return None
        
        self._current_points.append(point)
        
        # Check if annotation is complete
        if self._is_annotation_complete():
            annotation = self._create_annotation()
            self._annotations.append(annotation)
            self._current_points = []
            self._clear_temp_items()
            return annotation
        else:
            self._update_temp_display()
            return None
    
    def cancel_current(self) -> None:
        """Cancel the current annotation in progress."""
        self._current_points = []
        self._clear_temp_items()
    
    def remove_annotation(self, index: int) -> None:
        """Remove an annotation by index."""
        if 0 <= index < len(self._annotations):
            ann = self._annotations[index]
            for item in ann.graphics_items:
                self._scene.removeItem(item)
            del self._annotations[index]
    
    def clear_all(self) -> None:
        """Clear all annotations."""
        for ann in self._annotations:
            for item in ann.graphics_items:
                self._scene.removeItem(item)
        self._annotations = []
        self._clear_temp_items()
        self._current_points = []
    
    def get_annotations(self) -> List[Annotation]:
        """Get all annotations."""
        return self._annotations.copy()
    
    def _is_annotation_complete(self) -> bool:
        """Check if current annotation has enough points."""
        n = len(self._current_points)
        
        if self._current_tool == AnnotationTool.DISTANCE:
            return n >= 2
        elif self._current_tool == AnnotationTool.AREA:
            return n >= 3  # Minimum 3 points for area
        elif self._current_tool == AnnotationTool.ANGLE:
            return n >= 3  # 3 points for angle
        elif self._current_tool == AnnotationTool.POINT:
            return n >= 1
        
        return False
    
    def _create_annotation(self) -> Annotation:
        """Create annotation from current points."""
        color = self.COLORS.get(self._current_tool, QColor("#89b4fa"))
        
        annotation = Annotation(
            tool=self._current_tool,
            points=self._current_points.copy(),
            color=color
        )
        
        # Calculate measurement
        self._calculate_measurement(annotation)
        
        # Create graphics items
        self._create_graphics_items(annotation)
        
        return annotation
    
    def _calculate_measurement(self, annotation: Annotation) -> None:
        """Calculate measurement value for annotation."""
        points = annotation.points
        
        if annotation.tool == AnnotationTool.DISTANCE:
            # Euclidean distance
            p1, p2 = points[0], points[1]
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            annotation.pixel_value = math.sqrt(dx*dx + dy*dy)
            
            if self._pixel_spacing:
                # Convert to real units
                dx_mm = dx * self._pixel_spacing.column_spacing
                dy_mm = dy * self._pixel_spacing.row_spacing
                annotation.real_value = math.sqrt(dx_mm*dx_mm + dy_mm*dy_mm)
                annotation.unit = self._pixel_spacing.unit
            else:
                annotation.real_value = annotation.pixel_value
                annotation.unit = "px"
        
        elif annotation.tool == AnnotationTool.AREA:
            # Shoelace formula for polygon area
            n = len(points)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += points[i].x() * points[j].y()
                area -= points[j].x() * points[i].y()
            annotation.pixel_value = abs(area) / 2.0
            
            if self._pixel_spacing:
                # Convert to real units (area = pixels * spacing^2)
                spacing_sq = (self._pixel_spacing.row_spacing * 
                             self._pixel_spacing.column_spacing)
                annotation.real_value = annotation.pixel_value * spacing_sq
                annotation.unit = self._pixel_spacing.unit
            else:
                annotation.real_value = annotation.pixel_value
                annotation.unit = "px"
        
        elif annotation.tool == AnnotationTool.ANGLE:
            # Angle between three points (vertex at middle point)
            p1, p2, p3 = points[0], points[1], points[2]
            
            # Vectors from vertex (p2) to other points
            v1 = (p1.x() - p2.x(), p1.y() - p2.y())
            v2 = (p3.x() - p2.x(), p3.y() - p2.y())
            
            # Dot product and magnitudes
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]*v1[0] + v1[1]*v1[1])
            mag2 = math.sqrt(v2[0]*v2[0] + v2[1]*v2[1])
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
                annotation.real_value = math.degrees(math.acos(cos_angle))
            else:
                annotation.real_value = 0.0
            
            annotation.pixel_value = annotation.real_value
            annotation.unit = "°"
        
        elif annotation.tool == AnnotationTool.POINT:
            annotation.pixel_value = 0
            annotation.real_value = 0
            annotation.unit = ""
    
    def _create_graphics_items(self, annotation: Annotation) -> None:
        """Create graphics items for annotation."""
        pen = QPen(annotation.color, self._pen_width)
        pen.setCosmetic(True)  # Constant width regardless of zoom
        
        brush = QBrush(annotation.color)
        
        if annotation.tool == AnnotationTool.DISTANCE:
            # Line
            p1, p2 = annotation.points[0], annotation.points[1]
            line = self._scene.addLine(
                p1.x(), p1.y(), p2.x(), p2.y(), pen
            )
            annotation.graphics_items.append(line)
            
            # End points
            for p in [p1, p2]:
                marker = self._scene.addEllipse(
                    p.x() - 4, p.y() - 4, 8, 8, pen, brush
                )
                annotation.graphics_items.append(marker)
            
            # Label at midpoint
            mid_x = (p1.x() + p2.x()) / 2
            mid_y = (p1.y() + p2.y()) / 2
            self._add_label(annotation, mid_x, mid_y - 15)
        
        elif annotation.tool == AnnotationTool.AREA:
            # Polygon
            polygon = QPolygonF(annotation.points)
            
            fill_color = QColor(annotation.color)
            fill_color.setAlpha(50)
            fill_brush = QBrush(fill_color)
            
            poly_item = self._scene.addPolygon(polygon, pen, fill_brush)
            annotation.graphics_items.append(poly_item)
            
            # Vertex markers
            for p in annotation.points:
                marker = self._scene.addEllipse(
                    p.x() - 3, p.y() - 3, 6, 6, pen, brush
                )
                annotation.graphics_items.append(marker)
            
            # Label at centroid
            cx = sum(p.x() for p in annotation.points) / len(annotation.points)
            cy = sum(p.y() for p in annotation.points) / len(annotation.points)
            self._add_label(annotation, cx, cy)
        
        elif annotation.tool == AnnotationTool.ANGLE:
            # Two lines from vertex
            p1, p2, p3 = annotation.points
            
            line1 = self._scene.addLine(p2.x(), p2.y(), p1.x(), p1.y(), pen)
            line2 = self._scene.addLine(p2.x(), p2.y(), p3.x(), p3.y(), pen)
            annotation.graphics_items.extend([line1, line2])
            
            # Arc to show angle
            # (simplified - just show the vertex marker)
            marker = self._scene.addEllipse(
                p2.x() - 5, p2.y() - 5, 10, 10, pen, brush
            )
            annotation.graphics_items.append(marker)
            
            # Label near vertex
            self._add_label(annotation, p2.x() + 15, p2.y() - 15)
        
        elif annotation.tool == AnnotationTool.POINT:
            p = annotation.points[0]
            
            # Crosshair
            size = 10
            h_line = self._scene.addLine(
                p.x() - size, p.y(), p.x() + size, p.y(), pen
            )
            v_line = self._scene.addLine(
                p.x(), p.y() - size, p.x(), p.y() + size, pen
            )
            annotation.graphics_items.extend([h_line, v_line])
            
            # Label
            self._add_label(annotation, p.x() + 12, p.y() - 12)
    
    def _add_label(self, annotation: Annotation, x: float, y: float) -> None:
        """Add text label to annotation."""
        text = annotation.display_text
        
        text_item = self._scene.addText(text)
        text_item.setPos(x, y)
        text_item.setDefaultTextColor(annotation.color)
        
        font = QFont("Segoe UI", self._font_size)
        font.setBold(True)
        text_item.setFont(font)
        
        # Add background for readability
        text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        
        annotation.graphics_items.append(text_item)
    
    def _update_annotation_label(self, annotation: Annotation) -> None:
        """Update the label text for an annotation."""
        # Find and update text item
        for item in annotation.graphics_items:
            if isinstance(item, QGraphicsTextItem):
                item.setPlainText(annotation.display_text)
                break
    
    def _update_temp_display(self) -> None:
        """Update temporary display during annotation creation."""
        self._clear_temp_items()
        
        if not self._current_points:
            return
        
        color = self.COLORS.get(self._current_tool, QColor("#89b4fa"))
        pen = QPen(color, self._pen_width, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        
        if self._current_tool == AnnotationTool.DISTANCE:
            if len(self._current_points) == 1:
                # Show start point
                p = self._current_points[0]
                marker = self._scene.addEllipse(
                    p.x() - 4, p.y() - 4, 8, 8, pen
                )
                self._temp_items.append(marker)
        
        elif self._current_tool == AnnotationTool.AREA:
            # Show polygon in progress
            if len(self._current_points) >= 2:
                for i in range(len(self._current_points) - 1):
                    p1 = self._current_points[i]
                    p2 = self._current_points[i + 1]
                    line = self._scene.addLine(
                        p1.x(), p1.y(), p2.x(), p2.y(), pen
                    )
                    self._temp_items.append(line)
            
            # Show vertices
            for p in self._current_points:
                marker = self._scene.addEllipse(
                    p.x() - 3, p.y() - 3, 6, 6, pen
                )
                self._temp_items.append(marker)
        
        elif self._current_tool == AnnotationTool.ANGLE:
            # Show lines from vertex
            if len(self._current_points) >= 2:
                p1, p2 = self._current_points[0], self._current_points[1]
                line = self._scene.addLine(
                    p2.x(), p2.y(), p1.x(), p1.y(), pen
                )
                self._temp_items.append(line)
    
    def _clear_temp_items(self) -> None:
        """Clear temporary graphics items."""
        for item in self._temp_items:
            self._scene.removeItem(item)
        self._temp_items = []


class AnnotationToolbar(QWidget):
    """
    Toolbar widget for annotation tools.
    
    Provides buttons for selecting annotation tools and
    displays measurement results.
    """
    
    toolChanged = pyqtSignal(AnnotationTool)
    clearRequested = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the toolbar UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("Annotations")
        title.setObjectName("titleLabel")
        layout.addWidget(title)
        
        # Tool buttons
        tools_group = QGroupBox("Tools")
        tools_layout = QHBoxLayout(tools_group)
        tools_layout.setSpacing(4)
        
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        
        tools = [
            (AnnotationTool.NONE, "🖱️", "Select (no annotation)"),
            (AnnotationTool.DISTANCE, "📏", "Measure Distance"),
            (AnnotationTool.AREA, "⬡", "Measure Area"),
            (AnnotationTool.ANGLE, "📐", "Measure Angle"),
            (AnnotationTool.POINT, "📍", "Mark Point"),
        ]
        
        for tool, icon, tooltip in tools:
            btn = QToolButton()
            btn.setText(icon)
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setFixedSize(36, 36)
            btn.setProperty("tool", tool)
            
            if tool == AnnotationTool.NONE:
                btn.setChecked(True)
            
            self._button_group.addButton(btn)
            tools_layout.addWidget(btn)
        
        self._button_group.buttonClicked.connect(self._on_tool_clicked)
        
        tools_layout.addStretch()
        layout.addWidget(tools_group)
        
        # Measurements list
        measurements_group = QGroupBox("Measurements")
        measurements_layout = QVBoxLayout(measurements_group)
        
        self._measurements_list = QListWidget()
        self._measurements_list.setMaximumHeight(150)
        measurements_layout.addWidget(self._measurements_list)
        
        # Clear button
        clear_btn = QPushButton("Clear All")
        clear_btn.setObjectName("secondaryButton")
        clear_btn.clicked.connect(self.clearRequested.emit)
        measurements_layout.addWidget(clear_btn)
        
        layout.addWidget(measurements_group)
        
        # Instructions
        self._instructions = QLabel("Select a tool and click on the image")
        self._instructions.setObjectName("subtitleLabel")
        self._instructions.setWordWrap(True)
        layout.addWidget(self._instructions)
        
        layout.addStretch()
    
    def _on_tool_clicked(self, button: QToolButton) -> None:
        """Handle tool button click."""
        tool = button.property("tool")
        self.toolChanged.emit(tool)
        
        # Update instructions
        instructions = {
            AnnotationTool.NONE: "Select a tool to start annotating",
            AnnotationTool.DISTANCE: "Click two points to measure distance",
            AnnotationTool.AREA: "Click 3+ points, then double-click to close polygon",
            AnnotationTool.ANGLE: "Click 3 points: start, vertex, end",
            AnnotationTool.POINT: "Click to mark a point",
        }
        self._instructions.setText(instructions.get(tool, ""))
    
    def add_measurement(self, annotation: Annotation) -> None:
        """Add a measurement to the list."""
        item = QListWidgetItem(f"{annotation.tool.value}: {annotation.display_text}")
        self._measurements_list.addItem(item)
    
    def clear_measurements(self) -> None:
        """Clear the measurements list."""
        self._measurements_list.clear()
    
    def set_tool(self, tool: AnnotationTool) -> None:
        """Set the active tool."""
        for btn in self._button_group.buttons():
            if btn.property("tool") == tool:
                btn.setChecked(True)
                break
