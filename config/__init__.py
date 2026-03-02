"""
Configuration
=============
Simulation parameters and feature-toggle flags.

Usage::

    from config import Config, FeatureToggles
"""

from config.config import Config
from config.feature_toggles import FeatureToggles

__all__ = [
    "Config",
    "FeatureToggles",
]
