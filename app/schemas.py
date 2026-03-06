from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class DebugPayload(BaseModel):
    reason: Optional[str] = None
    qr_bbox: Optional[BoundingBox] = None
    roi_above_bbox: Optional[BoundingBox] = None
    roi_below_bbox: Optional[BoundingBox] = None
    orientation_angle: Optional[int] = None
    orientation_scores: Optional[list[str]] = None
    wb_attempts: Optional[list[str]] = None
    code_attempts: Optional[list[str]] = None
    code_uncertain_indices: Optional[list[int]] = None
    overlay_base64: Optional[str] = None
    ocr_provider: Optional[str] = None
    mistral_model: Optional[str] = None
    mistral_request_id: Optional[str] = None
    mistral_status_code: Optional[int] = None
    mistral_error: Optional[str] = None


class AnalyzeResponse(BaseModel):
    qr_found: bool
    qr_content: Optional[str] = None
    wb_above_qr: bool
    code_below_qr: Optional[str] = None
    confidence: float = Field(ge=0, le=1)
    debug: DebugPayload
