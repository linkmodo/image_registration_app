"""
Longitudinal series manager widget.

Provides management of multiple images in a longitudinal series
for sequential registration and comparison.
"""

from typing import Optional, List
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QGroupBox, QMenu, QFileDialog, QFrame
)
from PyQt6.QtGui import QIcon, QAction, QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QSize
import numpy as np

from ophthalmic_registration.core.image_data import ImageData


class SeriesItem:
    """Represents an image in the longitudinal series."""
    
    def __init__(
        self,
        image_data: ImageData,
        filepath: str,
        label: str = ""
    ):
        self.image_data = image_data
        self.filepath = filepath
        self.label = label or Path(filepath).stem
        self.acquisition_date = image_data.metadata.acquisition_date
        self.is_baseline = False
        self.is_registered = False
        self.registration_result = None
    
    @property
    def display_name(self) -> str:
        """Get display name for the item."""
        date_str = ""
        if self.acquisition_date:
            date_str = f" ({self.acquisition_date.strftime('%Y-%m-%d')})"
        
        status = ""
        if self.is_baseline:
            status = " [Baseline]"
        elif self.is_registered:
            status = " ✓"
        
        return f"{self.label}{date_str}{status}"


class SeriesManager(QWidget):
    """
    Widget for managing longitudinal image series.
    
    Supports loading multiple images, setting baseline, and
    tracking registration status.
    
    Signals:
        baselineChanged: Emitted when baseline selection changes
        selectionChanged: Emitted when current selection changes
        seriesUpdated: Emitted when series content changes
    """
    
    baselineChanged = pyqtSignal(object)  # SeriesItem
    selectionChanged = pyqtSignal(object)  # SeriesItem
    seriesUpdated = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._items: List[SeriesItem] = []
        self._baseline_index: int = -1
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the series manager UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Header
        header = QFrame()
        header.setStyleSheet("background-color: #181825; padding: 8px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        
        title = QLabel("Longitudinal Series")
        title.setObjectName("titleLabel")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        self._count_label = QLabel("0 images")
        self._count_label.setObjectName("subtitleLabel")
        header_layout.addWidget(self._count_label)
        
        layout.addWidget(header)
        
        # List widget
        self._list = QListWidget()
        self._list.setMinimumHeight(150)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.currentRowChanged.connect(self._on_selection_changed)
        self._list.setStyleSheet("""
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #313244;
            }
            QListWidget::item:selected {
                background-color: #45475a;
            }
        """)
        layout.addWidget(self._list, 1)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self._add_btn = QPushButton("+ Add")
        self._add_btn.setObjectName("secondaryButton")
        self._add_btn.clicked.connect(self._on_add_clicked)
        btn_layout.addWidget(self._add_btn)
        
        self._remove_btn = QPushButton("− Remove")
        self._remove_btn.setObjectName("secondaryButton")
        self._remove_btn.setEnabled(False)
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        btn_layout.addWidget(self._remove_btn)
        
        self._set_baseline_btn = QPushButton("Set as Baseline")
        self._set_baseline_btn.setObjectName("secondaryButton")
        self._set_baseline_btn.setEnabled(False)
        self._set_baseline_btn.clicked.connect(self._on_set_baseline_clicked)
        btn_layout.addWidget(self._set_baseline_btn)
        
        layout.addLayout(btn_layout)
    
    def add_image(
        self,
        image_data: ImageData,
        filepath: str,
        label: str = ""
    ) -> SeriesItem:
        """
        Add an image to the series.
        
        Args:
            image_data: ImageData object
            filepath: Original file path
            label: Optional display label
        
        Returns:
            Created SeriesItem
        """
        item = SeriesItem(image_data, filepath, label)
        self._items.append(item)
        
        # Set as baseline if first image
        if len(self._items) == 1:
            self._baseline_index = 0
            item.is_baseline = True
            self.baselineChanged.emit(item)
        
        self._update_list()
        self.seriesUpdated.emit()
        
        return item
    
    def remove_image(self, index: int) -> None:
        """Remove an image from the series."""
        if 0 <= index < len(self._items):
            was_baseline = self._items[index].is_baseline
            del self._items[index]
            
            if was_baseline:
                if self._items:
                    self._baseline_index = 0
                    self._items[0].is_baseline = True
                    self.baselineChanged.emit(self._items[0])
                else:
                    self._baseline_index = -1
            elif index < self._baseline_index:
                self._baseline_index -= 1
            
            self._update_list()
            self.seriesUpdated.emit()
    
    def set_baseline(self, index: int) -> None:
        """Set an image as the baseline."""
        if 0 <= index < len(self._items):
            # Clear previous baseline
            for item in self._items:
                item.is_baseline = False
            
            self._baseline_index = index
            self._items[index].is_baseline = True
            
            self._update_list()
            self.baselineChanged.emit(self._items[index])
    
    def get_baseline(self) -> Optional[SeriesItem]:
        """Get the baseline image."""
        if 0 <= self._baseline_index < len(self._items):
            return self._items[self._baseline_index]
        return None
    
    def get_current(self) -> Optional[SeriesItem]:
        """Get the currently selected image."""
        row = self._list.currentRow()
        if 0 <= row < len(self._items):
            return self._items[row]
        return None
    
    def get_all_items(self) -> List[SeriesItem]:
        """Get all items in the series."""
        return self._items.copy()
    
    def mark_registered(self, index: int, result) -> None:
        """Mark an image as registered."""
        if 0 <= index < len(self._items):
            self._items[index].is_registered = True
            self._items[index].registration_result = result
            self._update_list()
    
    def clear(self) -> None:
        """Clear all images from the series."""
        self._items.clear()
        self._baseline_index = -1
        self._update_list()
        self.seriesUpdated.emit()
    
    def _update_list(self) -> None:
        """Update the list widget display."""
        self._list.clear()
        
        for item in self._items:
            list_item = QListWidgetItem(item.display_name)
            
            if item.is_baseline:
                list_item.setForeground(Qt.GlobalColor.cyan)
            elif item.is_registered:
                list_item.setForeground(Qt.GlobalColor.green)
            
            self._list.addItem(list_item)
        
        self._count_label.setText(f"{len(self._items)} images")
        
        # Update button states
        has_selection = self._list.currentRow() >= 0
        self._remove_btn.setEnabled(has_selection)
        self._set_baseline_btn.setEnabled(has_selection)
    
    def _on_selection_changed(self, row: int) -> None:
        """Handle selection change."""
        self._remove_btn.setEnabled(row >= 0)
        self._set_baseline_btn.setEnabled(row >= 0)
        
        if 0 <= row < len(self._items):
            self.selectionChanged.emit(self._items[row])
    
    def _on_add_clicked(self) -> None:
        """Handle add button click."""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Images to Series",
            "",
            "Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.dcm);;All Files (*)"
        )
        
        if filepaths:
            from ophthalmic_registration.io.image_io import ImageLoader
            loader = ImageLoader()
            
            for filepath in filepaths:
                try:
                    image_data = loader.load(filepath)
                    self.add_image(image_data, filepath)
                except Exception as e:
                    print(f"Failed to load {filepath}: {e}")
    
    def _on_remove_clicked(self) -> None:
        """Handle remove button click."""
        row = self._list.currentRow()
        if row >= 0:
            self.remove_image(row)
    
    def _on_set_baseline_clicked(self) -> None:
        """Handle set baseline button click."""
        row = self._list.currentRow()
        if row >= 0:
            self.set_baseline(row)
    
    def _show_context_menu(self, pos) -> None:
        """Show context menu for list items."""
        item = self._list.itemAt(pos)
        if item is None:
            return
        
        row = self._list.row(item)
        
        menu = QMenu(self)
        
        set_baseline_action = QAction("Set as Baseline", self)
        set_baseline_action.triggered.connect(lambda: self.set_baseline(row))
        menu.addAction(set_baseline_action)
        
        menu.addSeparator()
        
        remove_action = QAction("Remove", self)
        remove_action.triggered.connect(lambda: self.remove_image(row))
        menu.addAction(remove_action)
        
        menu.exec(self._list.mapToGlobal(pos))
