"""
Application entry point for Ophthalmic Image Registration GUI.

Provides the main application runner and configuration.
"""

import sys
import logging
from typing import Optional, List

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtCore import Qt

from ophthalmic_registration.gui.main_window import MainWindow
from ophthalmic_registration.utils.logging_config import setup_logging


def run_application(
    args: Optional[List[str]] = None,
    log_level: str = "INFO"
) -> int:
    """
    Run the Ophthalmic Image Registration GUI application.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        log_level: Logging level
    
    Returns:
        Application exit code
    
    Example:
        >>> from ophthalmic_registration.gui import run_application
        >>> run_application()
    """
    # Setup logging
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Ophthalmic Image Registration GUI")
    
    # Enable high DPI scaling BEFORE creating QApplication
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create application
    if args is None:
        args = sys.argv
    
    app = QApplication(args)
    
    # Configure application
    app.setApplicationName("Ophthalmic Image Registration")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Medical Imaging")
    
    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    logger.info("Application window opened")
    
    # Run event loop
    exit_code = app.exec()
    
    logger.info(f"Application exited with code {exit_code}")
    
    return exit_code


def main():
    """Main entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ophthalmic Image Registration GUI"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Path to baseline image to load on startup"
    )
    parser.add_argument(
        "--followup",
        type=str,
        help="Path to follow-up image to load on startup"
    )
    
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.debug else "INFO"
    
    sys.exit(run_application(log_level=log_level))


if __name__ == "__main__":
    main()
