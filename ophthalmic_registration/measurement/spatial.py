"""
Spatial calibration and measurement utilities for ophthalmic images.

This module provides tools for converting pixel measurements to real-world
units, creating calibrated measurements, and generating scale overlays.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ophthalmic_registration.core.image_data import ImageData, PixelSpacing
from ophthalmic_registration.core.exceptions import CalibrationError

logger = logging.getLogger(__name__)


class MeasurementType(Enum):
    """Types of spatial measurements."""
    DISTANCE = "distance"
    AREA = "area"
    ANGLE = "angle"
    PERIMETER = "perimeter"


@dataclass
class Measurement:
    """
    A spatial measurement with both pixel and real-world values.
    
    Attributes:
        measurement_type: Type of measurement
        pixel_value: Value in pixels (or square pixels for area)
        real_value: Value in real-world units
        unit: Unit of real-world measurement
        points: List of (x, y) points defining the measurement
        label: Optional descriptive label
        metadata: Additional measurement metadata
    """
    measurement_type: MeasurementType
    pixel_value: float
    real_value: Optional[float] = None
    unit: str = "px"
    points: List[Tuple[float, float]] = field(default_factory=list)
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_calibrated(self) -> bool:
        """Check if measurement has real-world calibration."""
        return self.real_value is not None and self.unit != "px"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "measurement_type": self.measurement_type.value,
            "pixel_value": self.pixel_value,
            "real_value": self.real_value,
            "unit": self.unit,
            "points": self.points,
            "label": self.label,
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        if self.is_calibrated:
            return f"{self.real_value:.2f} {self.unit}"
        return f"{self.pixel_value:.2f} px"


class SpatialCalibration:
    """
    Spatial calibration and measurement tools for ophthalmic images.
    
    Provides methods for converting pixel distances to real-world units,
    creating measurements, and generating scale bar overlays.
    
    Attributes:
        pixel_spacing: Pixel spacing information
        default_unit: Default output unit for measurements
    
    Example:
        >>> calibration = SpatialCalibration(image_data.pixel_spacing)
        >>> 
        >>> # Measure distance between two points
        >>> distance = calibration.measure_distance((100, 100), (200, 200))
        >>> print(f"Distance: {distance}")
        >>> 
        >>> # Add scale bar to image
        >>> image_with_scale = calibration.add_scale_bar(image_data)
    """
    
    def __init__(
        self,
        pixel_spacing: Optional[PixelSpacing] = None,
        default_unit: str = "mm"
    ):
        """
        Initialize spatial calibration.
        
        Args:
            pixel_spacing: Pixel spacing information
            default_unit: Default unit for measurements ('mm' or 'um')
        """
        self.pixel_spacing = pixel_spacing
        self.default_unit = default_unit
    
    @classmethod
    def from_image(cls, image_data: ImageData, default_unit: str = "mm") -> 'SpatialCalibration':
        """
        Create calibration from ImageData.
        
        Args:
            image_data: Image with pixel spacing metadata
            default_unit: Default measurement unit
        
        Returns:
            SpatialCalibration instance
        """
        return cls(
            pixel_spacing=image_data.pixel_spacing,
            default_unit=default_unit
        )
    
    @property
    def is_calibrated(self) -> bool:
        """Check if calibration information is available."""
        return self.pixel_spacing is not None
    
    def pixels_to_mm(self, pixels: float, direction: str = "mean") -> float:
        """
        Convert pixels to millimeters.
        
        Args:
            pixels: Distance in pixels
            direction: 'row', 'column', or 'mean'
        
        Returns:
            Distance in millimeters
        
        Raises:
            CalibrationError: If pixel spacing is not available
        """
        if not self.is_calibrated:
            raise CalibrationError("Pixel spacing not available for calibration")
        
        if direction == "row":
            spacing = self.pixel_spacing.row_spacing
        elif direction == "column":
            spacing = self.pixel_spacing.column_spacing
        else:
            spacing = self.pixel_spacing.mean_spacing
        
        # Convert to mm if needed
        if self.pixel_spacing.unit == "um":
            spacing = spacing / 1000.0
        
        return pixels * spacing
    
    def pixels_to_microns(self, pixels: float, direction: str = "mean") -> float:
        """
        Convert pixels to microns.
        
        Args:
            pixels: Distance in pixels
            direction: 'row', 'column', or 'mean'
        
        Returns:
            Distance in microns
        """
        mm = self.pixels_to_mm(pixels, direction)
        return mm * 1000.0
    
    def mm_to_pixels(self, mm: float, direction: str = "mean") -> float:
        """
        Convert millimeters to pixels.
        
        Args:
            mm: Distance in millimeters
            direction: 'row', 'column', or 'mean'
        
        Returns:
            Distance in pixels
        """
        if not self.is_calibrated:
            raise CalibrationError("Pixel spacing not available for calibration")
        
        if direction == "row":
            spacing = self.pixel_spacing.row_spacing
        elif direction == "column":
            spacing = self.pixel_spacing.column_spacing
        else:
            spacing = self.pixel_spacing.mean_spacing
        
        if self.pixel_spacing.unit == "um":
            spacing = spacing / 1000.0
        
        return mm / spacing
    
    def measure_distance(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        unit: Optional[str] = None,
        label: Optional[str] = None
    ) -> Measurement:
        """
        Measure distance between two points.
        
        Args:
            point1: First point (x, y) in pixels
            point2: Second point (x, y) in pixels
            unit: Output unit ('mm', 'um', or 'px')
            label: Optional measurement label
        
        Returns:
            Measurement object with distance
        """
        unit = unit or self.default_unit
        
        # Calculate pixel distance
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        pixel_distance = np.sqrt(dx ** 2 + dy ** 2)
        
        # Convert to real units if calibrated
        real_distance = None
        output_unit = "px"
        
        if self.is_calibrated:
            try:
                if unit == "mm":
                    real_distance = self.pixels_to_mm(pixel_distance)
                    output_unit = "mm"
                elif unit == "um":
                    real_distance = self.pixels_to_microns(pixel_distance)
                    output_unit = "μm"
            except CalibrationError:
                logger.warning("Calibration failed, returning pixel distance")
        
        return Measurement(
            measurement_type=MeasurementType.DISTANCE,
            pixel_value=pixel_distance,
            real_value=real_distance,
            unit=output_unit,
            points=[point1, point2],
            label=label,
            metadata={
                "dx_pixels": dx,
                "dy_pixels": dy,
            }
        )
    
    def measure_area(
        self,
        points: List[Tuple[float, float]],
        unit: Optional[str] = None,
        label: Optional[str] = None
    ) -> Measurement:
        """
        Measure area of a polygon defined by points.
        
        Args:
            points: List of (x, y) vertices defining the polygon
            unit: Output unit ('mm', 'um', or 'px')
            label: Optional measurement label
        
        Returns:
            Measurement object with area
        """
        unit = unit or self.default_unit
        
        if len(points) < 3:
            raise ValueError("At least 3 points required for area measurement")
        
        # Calculate pixel area using shoelace formula
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        pixel_area = abs(area) / 2.0
        
        # Convert to real units if calibrated
        real_area = None
        output_unit = "px²"
        
        if self.is_calibrated:
            try:
                # Area conversion requires squaring the spacing
                spacing_mm = self.pixel_spacing.mean_spacing
                if self.pixel_spacing.unit == "um":
                    spacing_mm = spacing_mm / 1000.0
                
                if unit == "mm":
                    real_area = pixel_area * (spacing_mm ** 2)
                    output_unit = "mm²"
                elif unit == "um":
                    real_area = pixel_area * ((spacing_mm * 1000) ** 2)
                    output_unit = "μm²"
            except CalibrationError:
                logger.warning("Calibration failed, returning pixel area")
        
        return Measurement(
            measurement_type=MeasurementType.AREA,
            pixel_value=pixel_area,
            real_value=real_area,
            unit=output_unit,
            points=points,
            label=label,
        )
    
    def measure_perimeter(
        self,
        points: List[Tuple[float, float]],
        closed: bool = True,
        unit: Optional[str] = None,
        label: Optional[str] = None
    ) -> Measurement:
        """
        Measure perimeter of a polygon or polyline.
        
        Args:
            points: List of (x, y) vertices
            closed: If True, connects last point to first
            unit: Output unit ('mm', 'um', or 'px')
            label: Optional measurement label
        
        Returns:
            Measurement object with perimeter
        """
        unit = unit or self.default_unit
        
        if len(points) < 2:
            raise ValueError("At least 2 points required for perimeter")
        
        # Calculate pixel perimeter
        pixel_perimeter = 0.0
        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            pixel_perimeter += np.sqrt(dx ** 2 + dy ** 2)
        
        if closed:
            dx = points[0][0] - points[-1][0]
            dy = points[0][1] - points[-1][1]
            pixel_perimeter += np.sqrt(dx ** 2 + dy ** 2)
        
        # Convert to real units
        real_perimeter = None
        output_unit = "px"
        
        if self.is_calibrated:
            try:
                if unit == "mm":
                    real_perimeter = self.pixels_to_mm(pixel_perimeter)
                    output_unit = "mm"
                elif unit == "um":
                    real_perimeter = self.pixels_to_microns(pixel_perimeter)
                    output_unit = "μm"
            except CalibrationError:
                pass
        
        return Measurement(
            measurement_type=MeasurementType.PERIMETER,
            pixel_value=pixel_perimeter,
            real_value=real_perimeter,
            unit=output_unit,
            points=points,
            label=label,
            metadata={"closed": closed}
        )
    
    def measure_angle(
        self,
        vertex: Tuple[float, float],
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        label: Optional[str] = None
    ) -> Measurement:
        """
        Measure angle between two rays from a vertex.
        
        Args:
            vertex: Vertex point (x, y)
            point1: First ray endpoint (x, y)
            point2: Second ray endpoint (x, y)
            label: Optional measurement label
        
        Returns:
            Measurement object with angle in degrees
        """
        # Calculate vectors from vertex
        v1 = np.array([point1[0] - vertex[0], point1[1] - vertex[1]])
        v2 = np.array([point2[0] - vertex[0], point2[1] - vertex[1]])
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return Measurement(
            measurement_type=MeasurementType.ANGLE,
            pixel_value=angle_deg,
            real_value=angle_deg,  # Angles are scale-independent
            unit="°",
            points=[vertex, point1, point2],
            label=label,
        )
    
    def add_scale_bar(
        self,
        image: ImageData,
        length_mm: float = 1.0,
        position: str = "bottom_right",
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 3,
        font_scale: float = 0.5,
        margin: int = 20
    ) -> ImageData:
        """
        Add a scale bar overlay to an image.
        
        Args:
            image: Input image
            length_mm: Desired scale bar length in mm
            position: Position ('bottom_right', 'bottom_left', 'top_right', 'top_left')
            color: Scale bar color (B, G, R)
            thickness: Bar thickness in pixels
            font_scale: Font scale for label
            margin: Margin from image edge
        
        Returns:
            Image with scale bar overlay
        
        Raises:
            CalibrationError: If pixel spacing is not available
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for scale bar rendering")
        
        if not self.is_calibrated:
            raise CalibrationError("Cannot add scale bar without pixel spacing")
        
        # Calculate scale bar length in pixels
        bar_length_px = int(self.mm_to_pixels(length_mm))
        
        # Ensure minimum visible length
        bar_length_px = max(bar_length_px, 20)
        
        # Create output image
        result = image.copy()
        img = result.pixel_array.copy()
        
        # Convert to color if grayscale
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        h, w = img.shape[:2]
        
        # Calculate position
        bar_height = thickness
        if position == "bottom_right":
            x1 = w - margin - bar_length_px
            y1 = h - margin - bar_height
        elif position == "bottom_left":
            x1 = margin
            y1 = h - margin - bar_height
        elif position == "top_right":
            x1 = w - margin - bar_length_px
            y1 = margin
        elif position == "top_left":
            x1 = margin
            y1 = margin
        else:
            x1 = w - margin - bar_length_px
            y1 = h - margin - bar_height
        
        x2 = x1 + bar_length_px
        y2 = y1 + bar_height
        
        # Draw scale bar
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        
        # Add end caps
        cap_height = bar_height * 2
        cv2.line(img, (x1, y1 - cap_height // 2), (x1, y2 + cap_height // 2), color, thickness)
        cv2.line(img, (x2, y1 - cap_height // 2), (x2, y2 + cap_height // 2), color, thickness)
        
        # Add label
        if length_mm >= 1:
            label = f"{length_mm:.0f} mm"
        else:
            label = f"{length_mm * 1000:.0f} μm"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
        
        text_x = x1 + (bar_length_px - text_w) // 2
        text_y = y1 - 5
        
        # Draw text with background for visibility
        cv2.rectangle(
            img,
            (text_x - 2, text_y - text_h - 2),
            (text_x + text_w + 2, text_y + 2),
            (0, 0, 0),
            -1
        )
        cv2.putText(img, label, (text_x, text_y), font, font_scale, color, 1, cv2.LINE_AA)
        
        result.pixel_array = img
        result.preprocessing_history.append(f"scale_bar_{length_mm}mm")
        
        return result
    
    def add_grid_overlay(
        self,
        image: ImageData,
        spacing_mm: float = 1.0,
        color: Tuple[int, int, int] = (128, 128, 128),
        thickness: int = 1,
        alpha: float = 0.3
    ) -> ImageData:
        """
        Add a calibrated grid overlay to an image.
        
        Args:
            image: Input image
            spacing_mm: Grid spacing in mm
            color: Grid line color (B, G, R)
            thickness: Line thickness
            alpha: Grid transparency (0-1)
        
        Returns:
            Image with grid overlay
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for grid rendering")
        
        if not self.is_calibrated:
            raise CalibrationError("Cannot add calibrated grid without pixel spacing")
        
        # Calculate grid spacing in pixels
        spacing_px = int(self.mm_to_pixels(spacing_mm))
        spacing_px = max(spacing_px, 10)  # Minimum 10 pixels
        
        result = image.copy()
        img = result.pixel_array.copy()
        
        # Convert to color if needed
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        h, w = img.shape[:2]
        
        # Create grid overlay
        overlay = img.copy()
        
        # Vertical lines
        for x in range(0, w, spacing_px):
            cv2.line(overlay, (x, 0), (x, h), color, thickness)
        
        # Horizontal lines
        for y in range(0, h, spacing_px):
            cv2.line(overlay, (0, y), (w, y), color, thickness)
        
        # Blend overlay with original
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        result.pixel_array = img
        result.preprocessing_history.append(f"grid_overlay_{spacing_mm}mm")
        
        return result
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """Get calibration information as dictionary."""
        if not self.is_calibrated:
            return {"calibrated": False}
        
        return {
            "calibrated": True,
            "row_spacing": self.pixel_spacing.row_spacing,
            "column_spacing": self.pixel_spacing.column_spacing,
            "unit": self.pixel_spacing.unit,
            "source": self.pixel_spacing.source,
            "is_isotropic": self.pixel_spacing.is_isotropic,
            "mean_spacing": self.pixel_spacing.mean_spacing,
        }


def check_scale_compatibility(
    image1: ImageData,
    image2: ImageData,
    tolerance: float = 0.1
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if two images have compatible spatial calibration.
    
    Args:
        image1: First image
        image2: Second image
        tolerance: Allowed relative difference in pixel spacing
    
    Returns:
        Tuple of (compatible, details_dict)
    """
    details = {
        "image1_calibrated": image1.pixel_spacing is not None,
        "image2_calibrated": image2.pixel_spacing is not None,
    }
    
    if not details["image1_calibrated"] and not details["image2_calibrated"]:
        details["message"] = "Neither image has spatial calibration"
        return True, details  # Both uncalibrated is compatible
    
    if details["image1_calibrated"] != details["image2_calibrated"]:
        details["message"] = "Only one image has spatial calibration"
        details["compatible"] = False
        return False, details
    
    # Both calibrated - compare spacings
    spacing1 = image1.pixel_spacing.mean_spacing
    spacing2 = image2.pixel_spacing.mean_spacing
    
    details["image1_spacing"] = spacing1
    details["image2_spacing"] = spacing2
    details["spacing_ratio"] = max(spacing1, spacing2) / min(spacing1, spacing2)
    
    relative_diff = abs(spacing1 - spacing2) / max(spacing1, spacing2)
    details["relative_difference"] = relative_diff
    
    compatible = relative_diff <= tolerance
    details["compatible"] = compatible
    
    if compatible:
        details["message"] = "Pixel spacings are compatible"
    else:
        details["message"] = f"Pixel spacing difference ({relative_diff:.1%}) exceeds tolerance ({tolerance:.1%})"
    
    return compatible, details
