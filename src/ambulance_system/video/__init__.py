"""Video package: source abstraction and acquisition/inference pipeline."""

from ambulance_system.video.pipeline import FramePacket, PipelineStats, VideoPipeline
from ambulance_system.video.source import (
    CaptureVideoSource,
    FileVideoSource,
    RTSPVideoSource,
    SyntheticVideoSource,
    VideoSource,
    VideoSourceError,
    WebcamVideoSource,
    create_source,
    redact_url,
)

__all__ = [
    "CaptureVideoSource",
    "FileVideoSource",
    "FramePacket",
    "PipelineStats",
    "RTSPVideoSource",
    "SyntheticVideoSource",
    "VideoPipeline",
    "VideoSource",
    "VideoSourceError",
    "WebcamVideoSource",
    "create_source",
    "redact_url",
]
