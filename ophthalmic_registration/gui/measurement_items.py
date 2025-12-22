"""
Custom QGraphicsItem classes for interactive measurements.

Provides draggable, selectable measurement items for distance and area measurements.
"""

import math
from typing import List, Optional
from PyQt6.QtWidgets import (
    QGraphicsItem, QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsPolygonItem, QGraphicsTextItem, QGraphicsRectItem,
    QMenu, QGraphicsItemGroup
)
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QObject
from PyQt6.QtGui import QPen, QBrush, QColor, QFont, QPolygonF, QPainterPath

from ophthalmic_registration.core.image_data import PixelSpacing


class MeasurementSignals(QObject):
    """Signals for measurement items."""
    deleteRequested = pyqtSignal(object)  # Measurement item
    updated = pyqtSignal(object)  # Measurement item


class DraggablePoint(QGraphicsEllipseItem):
    """Draggable point for measurement endpoints."""
    
    def __init__(self, x: float, y: float, parent_measurement):
        super().__init__(x - 4, y - 4, 8, 8)
        self.parent_measurement = parent_measurement
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setPen(QPen(QColor("#f38ba8"), 2))
        self.setBrush(QBrush(QColor("#f38ba8")))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(100)
    
    def itemChange(self, change, value):
        """Handle item changes to update parent measurement."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if self.parent_measurement:
                self.parent_measurement.update_from_points()
        return super().itemChange(change, value)


class DistanceMeasurement(QGraphicsItemGroup):
    """Interactive distance measurement with draggable endpoints."""
    
    def __init__(self, p1: QPointF, p2: QPointF, pixel_spacing: Optional[PixelSpacing] = None):
        super().__init__()
        self.signals = MeasurementSignals()
        self.pixel_spacing = pixel_spacing
        
        # Create draggable points
        self.point1 = DraggablePoint(p1.x(), p1.y(), self)
        self.point2 = DraggablePoint(p2.x(), p2.y(), self)
        self.addToGroup(self.point1)
        self.addToGroup(self.point2)
        
        # Create line
        self.line = QGraphicsLineItem()
        self.line.setPen(QPen(QColor("#a6e3a1"), 2))
        self.addToGroup(self.line)
        
        # Create text label
        self.text = QGraphicsTextItem()
        self.text.setDefaultTextColor(QColor("#cdd6f4"))
        self.text.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.text.setZValue(101)
        self.addToGroup(self.text)
        
        # Create text background
        self.text_bg = QGraphicsRectItem()
        self.text_bg.setPen(QPen(Qt.PenStyle.NoPen))
        self.text_bg.setBrush(QBrush(QColor(0, 0, 0, 180)))
        self.text_bg.setZValue(100)
        self.addToGroup(self.text_bg)
        
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.RightButton | Qt.MouseButton.LeftButton)
        
        self.update_from_points()
    
    def update_from_points(self):
        """Update line and text based on current point positions."""
        p1_center = self.point1.rect().center() + self.point1.pos()
        p2_center = self.point2.rect().center() + self.point2.pos()
        
        # Update line
        self.line.setLine(p1_center.x(), p1_center.y(), p2_center.x(), p2_center.y())
        
        # Calculate distance
        dx = p2_center.x() - p1_center.x()
        dy = p2_center.y() - p1_center.y()
        pixel_distance = math.sqrt(dx * dx + dy * dy)
        
        # Convert to mm if pixel spacing available
        if self.pixel_spacing and self.pixel_spacing.row_spacing > 0:
            mm_distance = pixel_distance * self.pixel_spacing.row_spacing
            distance_text = f"{mm_distance:.2f} mm"
        else:
            distance_text = f"{pixel_distance:.1f} px"
        
        # Update text
        self.text.setPlainText(distance_text)
        
        # Position text at midpoint with offset perpendicular to line
        mid_x = (p1_center.x() + p2_center.x()) / 2
        mid_y = (p1_center.y() + p2_center.y()) / 2
        
        # Calculate perpendicular offset to avoid line overlap
        angle = math.atan2(dy, dx)
        perp_angle = angle + math.pi / 2
        offset_dist = 20  # pixels offset from line
        text_x = mid_x + offset_dist * math.cos(perp_angle)
        text_y = mid_y + offset_dist * math.sin(perp_angle)
        
        self.text.setPos(text_x, text_y)
        
        # Update background to match text position
        text_rect = self.text.boundingRect()
        self.text_bg.setRect(
            text_x - 2, text_y - 2,
            text_rect.width() + 4, text_rect.height() + 4
        )
        
        self.signals.updated.emit(self)
    
    def mousePressEvent(self, event):
        """Handle mouse press for context menu."""
        if event.button() == Qt.MouseButton.RightButton:
            menu = QMenu()
            delete_action = menu.addAction("Delete Measurement")
            action = menu.exec(event.screenPos())
            
            if action == delete_action:
                self.signals.deleteRequested.emit(self)
            event.accept()
        else:
            super().mousePressEvent(event)


class AreaMeasurement(QGraphicsItemGroup):
    """Interactive area measurement with editable polygon."""
    
    def __init__(self, points: List[QPointF], pixel_spacing: Optional[PixelSpacing] = None):
        super().__init__()
        self.signals = MeasurementSignals()
        self.pixel_spacing = pixel_spacing
        self.draggable_points: List[DraggablePoint] = []
        
        # Create draggable points for each vertex
        for pt in points:
            point = DraggablePoint(pt.x(), pt.y(), self)
            self.draggable_points.append(point)
            self.addToGroup(point)
        
        # Create polygon
        self.polygon = QGraphicsPolygonItem()
        self.polygon.setPen(QPen(QColor("#89b4fa"), 2))
        self.polygon.setBrush(QBrush(QColor(137, 180, 250, 50)))
        self.addToGroup(self.polygon)
        
        # Create text label
        self.text = QGraphicsTextItem()
        self.text.setDefaultTextColor(QColor("#cdd6f4"))
        self.text.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.text.setZValue(101)
        self.addToGroup(self.text)
        
        # Create text background
        self.text_bg = QGraphicsRectItem()
        self.text_bg.setPen(QPen(Qt.PenStyle.NoPen))
        self.text_bg.setBrush(QBrush(QColor(0, 0, 0, 180)))
        self.text_bg.setZValue(100)
        self.addToGroup(self.text_bg)
        
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.RightButton | Qt.MouseButton.LeftButton)
        
        self.update_from_points()
    
    def update_from_points(self):
        """Update polygon and text based on current point positions."""
        # Get current point positions
        points = []
        for pt in self.draggable_points:
            center = pt.rect().center() + pt.pos()
            points.append(center)
        
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
        
        # Convert to mm²
        if self.pixel_spacing and self.pixel_spacing.row_spacing > 0 and self.pixel_spacing.column_spacing > 0:
            area_mm2 = area_pixels * self.pixel_spacing.row_spacing * self.pixel_spacing.column_spacing
            area_text = f"{area_mm2:.2f} mm²"
        else:
            area_text = f"{area_pixels:.0f} px²"
        
        # Update text
        self.text.setPlainText(area_text)
        
        # Position text at centroid with offset to avoid polygon overlap
        cx = sum(p.x() for p in points) / n
        cy = sum(p.y() for p in points) / n
        
        # Offset text above centroid
        text_offset_y = -25  # pixels above centroid
        text_x = cx
        text_y = cy + text_offset_y
        
        self.text.setPos(text_x, text_y)
        
        # Update background to match text position
        text_rect = self.text.boundingRect()
        self.text_bg.setRect(
            text_x - 2, text_y - 2,
            text_rect.width() + 4, text_rect.height() + 4
        )
        
        self.signals.updated.emit(self)
    
    def mousePressEvent(self, event):
        """Handle mouse press for context menu."""
        if event.button() == Qt.MouseButton.RightButton:
            menu = QMenu()
            delete_action = menu.addAction("Delete Measurement")
            action = menu.exec(event.screenPos())
            
            if action == delete_action:
                self.signals.deleteRequested.emit(self)
            event.accept()
        else:
            super().mousePressEvent(event)
