"""Bubblify - Interactive URDF spherization tool using Viser.

This package provides tools for creating and editing primitive-based collision
representations of robot URDFs using an interactive Viser-based GUI.
"""

from .core import Primitive, PrimitiveStore, EnhancedViserUrdf
from .gui import BubblifyApp

__all__ = ["Primitive", "PrimitiveStore", "EnhancedViserUrdf", "BubblifyApp"]
__version__ = "0.1.0"
