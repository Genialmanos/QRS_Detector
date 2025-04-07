"""This script allows users to import directly the most useful functions."""

# Import main functions from the module
from .fast_qrs_detector import qrs_detector, detect_peaks, threshold_detection, preprocess_ecg
from .print_qrs import print_signal_with_qrs

# Define what should be available when using "from fast_qrs_detector import *"
__all__ = [
    'qrs_detector',
    'detect_peaks',
    'threshold_detection',
    'preprocess_ecg',
    'print_signal_with_qrs'
]
