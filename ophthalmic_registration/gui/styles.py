"""
Modern styling for the Ophthalmic Registration GUI.

Provides a clean, professional dark/light theme suitable for
medical imaging applications.
"""

DARK_THEME = """
QMainWindow {
    background-color: #1e1e2e;
}

QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 10pt;
}

QMenuBar {
    background-color: #181825;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
    padding: 4px;
}

QMenuBar::item:selected {
    background-color: #45475a;
    border-radius: 4px;
}

QMenu {
    background-color: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 4px;
}

QMenu::item {
    padding: 8px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #45475a;
}

QToolBar {
    background-color: #181825;
    border: none;
    spacing: 8px;
    padding: 8px;
}

QToolButton {
    background-color: transparent;
    border: none;
    border-radius: 6px;
    padding: 8px;
    color: #cdd6f4;
}

QToolButton:hover {
    background-color: #45475a;
}

QToolButton:pressed {
    background-color: #585b70;
}

QToolButton:checked {
    background-color: #89b4fa;
    color: #1e1e2e;
}

QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #b4befe;
}

QPushButton:pressed {
    background-color: #74c7ec;
}

QPushButton:disabled {
    background-color: #45475a;
    color: #6c7086;
}

QPushButton#secondaryButton {
    background-color: #45475a;
    color: #cdd6f4;
}

QPushButton#secondaryButton:hover {
    background-color: #585b70;
}

QPushButton#dangerButton {
    background-color: #f38ba8;
    color: #1e1e2e;
}

QLabel {
    color: #cdd6f4;
    background-color: transparent;
}

QLabel#titleLabel {
    font-size: 14pt;
    font-weight: bold;
    color: #89b4fa;
}

QLabel#subtitleLabel {
    font-size: 9pt;
    color: #a6adc8;
}

QGroupBox {
    border: 1px solid #313244;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    background-color: #181825;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    color: #89b4fa;
    font-weight: bold;
}

QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px 12px;
    color: #cdd6f4;
    min-width: 120px;
}

QComboBox:hover {
    border-color: #89b4fa;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #cdd6f4;
    margin-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 6px;
    selection-background-color: #45475a;
}

QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px;
    color: #cdd6f4;
}

QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #89b4fa;
}

QSlider::groove:horizontal {
    height: 6px;
    background-color: #313244;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background-color: #89b4fa;
    width: 18px;
    height: 18px;
    margin: -6px 0;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background-color: #b4befe;
}

QSlider::sub-page:horizontal {
    background-color: #89b4fa;
    border-radius: 3px;
}

QProgressBar {
    background-color: #313244;
    border: none;
    border-radius: 6px;
    height: 8px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 6px;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #181825;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #585b70;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #181825;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #45475a;
    border-radius: 6px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #585b70;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

QTabWidget::pane {
    border: 1px solid #313244;
    border-radius: 8px;
    background-color: #181825;
}

QTabBar::tab {
    background-color: #1e1e2e;
    color: #a6adc8;
    padding: 10px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #181825;
    color: #89b4fa;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background-color: #313244;
}

QListWidget {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 4px;
}

QListWidget::item {
    padding: 8px;
    border-radius: 4px;
}

QListWidget::item:selected {
    background-color: #45475a;
}

QListWidget::item:hover:!selected {
    background-color: #313244;
}

QStatusBar {
    background-color: #181825;
    color: #a6adc8;
    border-top: 1px solid #313244;
}

QSplitter::handle {
    background-color: #313244;
}

QSplitter::handle:hover {
    background-color: #89b4fa;
}

QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    border: 2px solid #45475a;
    background-color: #313244;
}

QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}

QCheckBox::indicator:hover {
    border-color: #89b4fa;
}

QFrame#separator {
    background-color: #313244;
    max-height: 1px;
}

QFrame#imagePanel {
    background-color: #11111b;
    border: 2px dashed #45475a;
    border-radius: 12px;
}

QFrame#imagePanel:hover {
    border-color: #89b4fa;
}

QFrame#imagePanelLoaded {
    background-color: #11111b;
    border: 2px solid #45475a;
    border-radius: 12px;
}
"""

LIGHT_THEME = """
QMainWindow {
    background-color: #eff1f5;
}

QWidget {
    background-color: #eff1f5;
    color: #4c4f69;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 10pt;
}

QMenuBar {
    background-color: #e6e9ef;
    color: #4c4f69;
    border-bottom: 1px solid #ccd0da;
    padding: 4px;
}

QMenuBar::item:selected {
    background-color: #ccd0da;
    border-radius: 4px;
}

QMenu {
    background-color: #eff1f5;
    border: 1px solid #ccd0da;
    border-radius: 8px;
    padding: 4px;
}

QMenu::item {
    padding: 8px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #ccd0da;
}

QPushButton {
    background-color: #1e66f5;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #2a6ff7;
}

QPushButton:pressed {
    background-color: #1a5ce0;
}

QPushButton:disabled {
    background-color: #ccd0da;
    color: #9ca0b0;
}

QLabel#titleLabel {
    font-size: 14pt;
    font-weight: bold;
    color: #1e66f5;
}

QGroupBox {
    border: 1px solid #ccd0da;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    background-color: #e6e9ef;
}

QGroupBox::title {
    color: #1e66f5;
    font-weight: bold;
}

QComboBox {
    background-color: #e6e9ef;
    border: 1px solid #ccd0da;
    border-radius: 6px;
    padding: 8px 12px;
    color: #4c4f69;
}

QComboBox:hover {
    border-color: #1e66f5;
}

QFrame#imagePanel {
    background-color: #dce0e8;
    border: 2px dashed #9ca0b0;
    border-radius: 12px;
}

QFrame#imagePanel:hover {
    border-color: #1e66f5;
}

QFrame#imagePanelLoaded {
    background-color: #dce0e8;
    border: 2px solid #9ca0b0;
    border-radius: 12px;
}
"""


def get_theme(dark: bool = True) -> str:
    """Get the stylesheet for the specified theme."""
    return DARK_THEME if dark else LIGHT_THEME
