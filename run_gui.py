#!/usr/bin/env python
"""
Launch script for Ophthalmic Image Registration GUI.

Usage:
    python run_gui.py
    python run_gui.py --debug
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from ophthalmic_registration.gui.app import main

if __name__ == "__main__":
    main()
