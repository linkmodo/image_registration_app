"""
Custom QGraphicsItem classes for interactive measurements.

Provides draggable, selectable measurement items for distance and area measurements.
"""

import math
from typing import List, Optional
from PyQt6.QtWidgets import (
    QGraphicsItem, QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsPolygonItem, QGraphicsTextItem, QGraphicsRectItem,
    QMenu, QGraphicsScene
)
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QObject
from PyQt6.QtGui import QPen, QBrush, QColor, QFont, QPolygonF

from ophthalmic_registration.core.image_data import PixelSpacing


class MeasurementSignals(QObject):
    """Signals for measurement items."""
    deleteRequested = pyqtSignal(object)  # Measurement item
    updated = pyqtSignal(object)  # Measurement item


class DraggablePoint(QGraphicsEllipseItem):
    """Draggable point for measurement endpoints."""
    
    def __init__(self, x: float, y: float, update_callback, delete_callback):
        super().__init__(-5, -5, 10, 10)
        self.setPos(x, y)
        self._update_callback = update_callback
        self._delete_callback = delete_callback
        
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setPen(QPen(QColor("#f38ba8"), 2))
        self.setBrush(QBrush(QColor("#f38ba8")))
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setZValue(100)
        self.setAcceptHoverEvents(True)
    
    def itemChange(self, change, value):
        """Handle item changes to update parent measurement."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if self._update_callback:
                self._update_callback()
        return super().itemChange(change, value)
    
    def hoverEnterEvent(self, event):
        """Highlight on hover."""
        self.setBrush(QBrush(QColor("#fab387")))
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Remove highlight."""
        self.setBrush(QBrush(QColor("#f38ba8")))
        super().hoverLeaveEvent(event)
    
    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.RightButton:
            menu = QMenu()
            delete_action = menu.addAction("Delete Measurement")
            action = menu.exec(event.screenPos())
            if action == delete_action:
                self._delete_callback()
            event.accept()
        else:
            super().mousePressEvent(event)


class DistanceMeasurement:
    """Interactive distance measurement with draggable endpoints."""
    
    # Class variable for unit preference
    use_mm_units = True  # Default to mm when available
    
    def __init__(self, scene: QGraphicsScene, p1: QPointF, p2: QPointF, 
                 pixel_spacing: Optional[PixelSpacing] = None):
        self.scene = scene
        self.signals = MeasurementSignals()
        self.pixel_spacing = pixel_spacing
        self._items = []  # Track all items for deletion
        
        # Create draggable points
        self.point1 = DraggablePoint(p1.x(), p1.y(), self.update_from_points, self._request_delete)
        self.point2 = DraggablePoint(p2.x(), p2.y(), self.update_from_points, self._request_delete)
        scene.addItem(self.point1)
        scene.addItem(self.point2)
        self._items.extend([self.point1, self.point2])
        
        # Create line
        self.line = QGraphicsLineItem()
        self.line.setPen(QPen(QColor("#a6e3a1"), 2))
        self.line.setZValue(50)
        scene.addItem(self.line)
        self._items.append(self.line)
        
        # Create text background
        self.text_bg = QGraphicsRectItem()
        self.text_bg.setPen(QPen(Qt.PenStyle.NoPen))
        self.text_bg.setBrush(QBrush(QColor(0, 0, 0, 180)))
        self.text_bg.setZValue(99)
        scene.addItem(self.text_bg)
        self._items.append(self.text_bg)
        
        # Create text label
        self.text = QGraphicsTextItem()
        self.text.setDefaultTextColor(QColor("#cdd6f4"))
        self.text.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.text.setZValue(101)
        scene.addItem(self.text)
        self._items.append(self.text)
        
        self.update_from_points()
    
    def _request_delete(self):
        """Request deletion through signal."""
        self.signals.deleteRequested.emit(self)
    
    def update_from_points(self):
        """Update line and text based on current point positions."""
        p1 = self.point1.pos()
        p2 = self.point2.pos()
        
        # Update line
        self.line.setLine(p1.x(), p1.y(), p2.x(), p2.y())
        
        # Calculate distance
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        pixel_distance = math.sqrt(dx * dx + dy * dy)
        
        # Convert to mm if pixel spacing available and user prefers mm
        if DistanceMeasurement.use_mm_units and self.pixel_spacing and self.pixel_spacing.row_spacing > 0:
            mm_distance = pixel_distance * self.pixel_spacing.row_spacing
            distance_text = f"{mm_distance:.2f} mm"
        else:
            distance_text = f"{pixel_distance:.1f} px"
        
        # Update text
        self.text.setPlainText(distance_text)
        
        # Position text above the midpoint (always offset upward)
        mid_x = (p1.x() + p2.x()) / 2
        mid_y = (p1.y() + p2.y()) / 2
        
        # Always offset text above the line
        text_x = mid_x - 20  # Slightly left of center
        text_y = mid_y - 25  # Above the line
        
        self.text.setPos(text_x, text_y)
        
        # Update background to match text position
        text_rect = self.text.boundingRect()
        self.text_bg.setRect(text_x - 2, text_y - 2, 
                             text_rect.width() + 4, text_rect.height() + 4)
        
        self.signals.updated.emit(self)
    
    def remove_from_scene(self):
        """Remove all items from scene."""
        for item in self._items:
            if item.scene():
                self.scene.removeItem(item)
        self._items.clear()
    
    def contains_item(self, item) -> bool:
        """Check if item belongs to this measurement."""
        return item in self._items


class ClickablePolygon(QGraphicsPolygonItem):
    """Polygon that can be right-clicked to delete."""
    
    def __init__(self, delete_callback):
        super().__init__()
        self._delete_callback = delete_callback
        self.setAcceptHoverEvents(True)
    
    def mousePressEvent(self, event):
        """Handle mouse press for context menu."""
        if event.button() == Qt.MouseButton.RightButton:
            menu = QMenu()
            delete_action = menu.addAction("Delete Measurement")
            action = menu.exec(event.screenPos())
            if action == delete_action:
                self._delete_callback()
            event.accept()
        else:
            super().mousePressEvent(event)


class AreaMeasurement:
    """Interactive area measurement with editable polygon (no visible dots)."""
    
    # Class variable for unit preference
    use_mm_units = True  # Default to mm when available
    
    def __init__(self, scene: QGraphicsScene, points: List[QPointF], 
                 pixel_spacing: Optional[PixelSpacing] = None):
        self.scene = scene
        self.signals = MeasurementSignals()
        self.pixel_spacing = pixel_spacing
        self._items = []  # Track all items for deletion
        self._vertex_positions: List[QPointF] = [QPointF(pt.x(), pt.y()) for pt in points]
        
        # Create clickable polygon (no visible dots, just the polygon outline)
        self.polygon = ClickablePolygon(self._request_delete)
        self.polygon.setPen(QPen(QColor("#89b4fa"), 2))
        self.polygon.setBrush(QBrush(QColor(137, 180, 250, 50)))
        self.polygon.setZValue(50)
        self.polygon.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        scene.addItem(self.polygon)
        self._items.append(self.polygon)
        
        # Create text background
        self.text_bg = QGraphicsRectItem()
        self.text_bg.setPen(QPen(Qt.PenStyle.NoPen))
        self.text_bg.setBrush(QBrush(QColor(0, 0, 0, 180)))
        self.text_bg.setZValue(99)
        scene.addItem(self.text_bg)
        self._items.append(self.text_bg)
        
        # Create text label
        self.text = QGraphicsTextItem()
        self.text.setDefaultTextColor(QColor("#cdd6f4"))
        self.text.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.text.setZValue(101)
        scene.addItem(self.text)
        self._items.append(self.text)
        
        self.update_from_points()
    
    def _request_delete(self):
        """Request deletion through signal."""
        self.signals.deleteRequested.emit(self)
    
    def update_from_points(self):
        """Update polygon and text based on current point positions."""
        # Get current point positions
        points = self._vertex_positions
        
        if len(points) < 3:
            return
        
        # Create closed polygon
        poly = QPolygonF()
        for pt in points:
            poly.append(pt)
        poly.append(points[0])  # Close the polygon
        
        self.polygon.setPolygon(poly)
        
        # Calculate area using shoelace formula
        n = len(points)
        area_pixels = 0.0
        for i in range(n):
            j = (i + 1) % n
            area_pixels += points[i].x() * points[j].y()
            area_pixels -= points[j].x() * points[i].y()
        area_pixels = abs(area_pixels) / 2.0
        
        # Convert to mm² if user prefers mm and pixel spacing available
        if AreaMeasurement.use_mm_units and self.pixel_spacing and self.pixel_spacing.row_spacing > 0 and self.pixel_spacing.column_spacing > 0:
            area_mm2 = area_pixels * self.pixel_spacing.row_spacing * self.pixel_spacing.column_spacing
            area_text = f"{area_mm2:.2f} mm²"
        else:
            area_text = f"{area_pixels:.0f} px²"
        
        # Update text
        self.text.setPlainText(area_text)
        
        # Position text at centroid with offset
        cx = sum(p.x() for p in points) / n
        cy = sum(p.y() for p in points) / n
        
        text_x = cx
        text_y = cy - 25
        
        self.text.setPos(text_x, text_y)
        
        # Update background to match text position
        text_rect = self.text.boundingRect()
        self.text_bg.setRect(text_x - 2, text_y - 2,
                             text_rect.width() + 4, text_rect.height() + 4)
        
        self.signals.updated.emit(self)
    
    def remove_from_scene(self):
        """Remove all items from scene."""
        for item in self._items:
            if item.scene():
                self.scene.removeItem(item)
        self._items.clear()
        self._vertex_positions.clear()
    
    def contains_item(self, item) -> bool:
        """Check if item belongs to this measurement."""
        return item in self._items or item == self.polygon
