"""
Logging configuration for ophthalmic image registration.

This module provides standardized logging setup for the application,
with configurable verbosity levels and output formats suitable for
both development and clinical/research environments.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


# Default format for log messages
DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)

# Simplified format for console output
CONSOLE_FORMAT = "%(levelname)-8s | %(message)s"

# Detailed format for file logging
FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
)


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for console output.
    
    Adds ANSI color codes to log levels for better visibility
    in terminal output.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, fmt: str = CONSOLE_FORMAT, use_colors: bool = True):
        super().__init__(fmt)
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors and record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        return super().format(record)


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    use_colors: bool = True,
    module_name: str = "ophthalmic_registration"
) -> logging.Logger:
    """
    Set up logging for the ophthalmic registration application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console_output: Enable console output
        use_colors: Use colored output in console
        module_name: Root module name for logger
    
    Returns:
        Configured root logger
    
    Example:
        >>> setup_logging(level=logging.DEBUG, log_file="registration.log")
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting registration...")
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Get the root logger for our module
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if use_colors and sys.stdout.isatty():
            formatter = ColoredFormatter(CONSOLE_FORMAT, use_colors=True)
        else:
            formatter = logging.Formatter(CONSOLE_FORMAT)
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing image...")
    """
    return logging.getLogger(name)


def create_session_log(
    output_dir: Union[str, Path],
    prefix: str = "registration"
) -> Path:
    """
    Create a timestamped log file for a registration session.
    
    Args:
        output_dir: Directory for log file
        prefix: Log file prefix
    
    Returns:
        Path to created log file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"{prefix}_{timestamp}.log"
    
    return log_file


class RegistrationLogger:
    """
    Specialized logger for registration operations.
    
    Provides structured logging with context tracking for
    multi-step registration workflows.
    
    Example:
        >>> reg_logger = RegistrationLogger("case_001")
        >>> reg_logger.start_registration()
        >>> reg_logger.log_coarse_alignment(num_matches=150, inliers=120)
        >>> reg_logger.log_fine_alignment(ecc=0.95, converged=True)
        >>> reg_logger.end_registration()
    """
    
    def __init__(
        self,
        session_id: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize registration logger.
        
        Args:
            session_id: Unique session identifier
            logger: Base logger (creates new if None)
        """
        self.session_id = session_id
        self.logger = logger or get_logger("ophthalmic_registration.session")
        self.start_time: Optional[datetime] = None
    
    def _log(self, level: int, message: str, **kwargs):
        """Log message with session context."""
        extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        full_message = f"[{self.session_id}] {message}"
        if extra_info:
            full_message += f" | {extra_info}"
        self.logger.log(level, full_message)
    
    def start_registration(self):
        """Log registration start."""
        self.start_time = datetime.now()
        self._log(logging.INFO, "Starting registration")
    
    def log_image_loaded(self, image_type: str, shape: tuple, has_spacing: bool):
        """Log image loading."""
        self._log(
            logging.INFO,
            f"Loaded {image_type} image",
            shape=shape,
            has_spacing=has_spacing
        )
    
    def log_preprocessing(self, steps: list):
        """Log preprocessing completion."""
        self._log(
            logging.INFO,
            f"Preprocessing complete",
            steps=len(steps)
        )
    
    def log_coarse_alignment(
        self,
        num_keypoints: int,
        num_matches: int,
        num_inliers: int
    ):
        """Log coarse alignment results."""
        self._log(
            logging.INFO,
            "Coarse alignment complete",
            keypoints=num_keypoints,
            matches=num_matches,
            inliers=num_inliers
        )
    
    def log_fine_alignment(self, ecc: float, converged: bool):
        """Log fine alignment results."""
        self._log(
            logging.INFO,
            "Fine alignment complete",
            ecc=f"{ecc:.4f}",
            converged=converged
        )
    
    def log_warning(self, message: str):
        """Log warning."""
        self._log(logging.WARNING, message)
    
    def log_error(self, message: str, exception: Optional[Exception] = None):
        """Log error."""
        if exception:
            self._log(logging.ERROR, f"{message}: {exception}")
        else:
            self._log(logging.ERROR, message)
    
    def end_registration(self, success: bool = True):
        """Log registration end."""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            self._log(
                logging.INFO if success else logging.ERROR,
                "Registration " + ("complete" if success else "failed"),
                duration_sec=f"{duration:.2f}"
            )
        else:
            self._log(
                logging.INFO if success else logging.ERROR,
                "Registration " + ("complete" if success else "failed")
            )
