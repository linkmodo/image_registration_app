"""
GUI module for Ophthalmic Image Registration.

Provides a modern PyQt6-based graphical user interface for
longitudinal ophthalmic image registration and comparison.
"""

from ophthalmic_registration.gui.main_window import MainWindow
from ophthalmic_registration.gui.app import run_application
from ophthalmic_registration.gui.batch_registration import BatchRegistrationDialog
from ophthalmic_registration.gui.annotation_tools import (
    AnnotationManager,
    AnnotationToolbar,
    AnnotationTool,
)
from ophthalmic_registration.gui.manual_registration import ManualRegistrationDialog

__all__ = [
    "MainWindow",
    "run_application",
    "BatchRegistrationDialog",
    "AnnotationManager",
    "AnnotationToolbar",
    "AnnotationTool",
    "ManualRegistrationDialog",
]
