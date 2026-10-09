"""Inference model wrappers (SCRFD detection, ArcFace embedding)."""

from __future__ import annotations

from campus.models.arcface import ArcFaceEmbedder, EmbedderConfig, EnrollmentQuality
from campus.models.scrfd import DetectorConfig, InsufficientFacesError, ScrfdDetector

__all__ = [
    "ArcFaceEmbedder",
    "DetectorConfig",
    "EmbedderConfig",
    "EnrollmentQuality",
    "InsufficientFacesError",
    "ScrfdDetector",
]
