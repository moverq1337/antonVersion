from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.config import settings
from app.schemas import AnalyzeResponse, BoundingBox, DebugPayload
from app.services.ocr import OCRProvider, OCRResult


@dataclass
class QRDetection:
    points: np.ndarray
    bbox: BoundingBox
    content: Optional[str] = None



@dataclass
class OrientationCandidate:
    angle: int
    score: float
    detection: QRDetection
    rectified: np.ndarray
    above_roi: np.ndarray
    below_roi: np.ndarray
    above_bbox: BoundingBox
    below_bbox: BoundingBox


@dataclass
class LineRead:
    text: str
    confidence: float
    uncertain_indices: list[int]

class ImageAnalyzer:
    def __init__(self, primary_ocr: OCRProvider, fallback_ocr: Optional[OCRProvider] = None):
        self.primary_ocr = primary_ocr
        self.fallback_ocr = fallback_ocr
        self.qr_detector = cv2.QRCodeDetector()

    def analyze(self, image: np.ndarray, require_wb: bool = True) -> AnalyzeResponse:
        try:
            oriented, orientation_scores = self._select_best_orientation(image)
            if oriented is None:
                return AnalyzeResponse(
                    qr_found=False,
                    qr_content=None,
                    wb_above_qr=False,
                    code_below_qr=None,
                    confidence=0.0,
                    debug=DebugPayload(reason='QR code not found', orientation_scores=orientation_scores, **self._provider_debug_payload()),
                )

            wb, wb_attempts = self._detect_wb_above_qr(oriented.above_roi)
            wb_ok = self._is_wb_match(wb.text)

            code, code_attempts = self._detect_code_below_qr(oriented.below_roi)

            if len(code.text) < 4:
                code = LineRead(text='', confidence=0.0, uncertain_indices=[])

            if require_wb:
                accepted_code = code.text if wb_ok and code.text else None
                if not wb_ok:
                    reason = f"WB not confirmed above QR (detected='{wb.text or '-'}')"
                elif not code.text:
                    reason = 'No alphanumeric code detected below QR'
                else:
                    reason = None
                confidence = max(0.0, min(1.0, (wb.confidence + code.confidence) / 2.0)) if wb_ok else wb.confidence * 0.5
            else:
                accepted_code = code.text if code.text else None
                reason = None if accepted_code else 'No alphanumeric code detected below QR'
                confidence = code.confidence
                wb_ok = True

            overlay = self._build_overlay(oriented.rectified) if settings.debug_overlay_enabled else ''
            return AnalyzeResponse(
                qr_found=True,
                qr_content=oriented.detection.content,
                wb_above_qr=wb_ok,
                code_below_qr=accepted_code,
                confidence=round(max(0.0, min(1.0, confidence)), 4),
                debug=DebugPayload(
                    reason=reason,
                    qr_bbox=oriented.detection.bbox,
                    roi_above_bbox=oriented.above_bbox,
                    roi_below_bbox=oriented.below_bbox,
                    orientation_angle=oriented.angle,
                    orientation_scores=orientation_scores,
                    wb_attempts=wb_attempts[:12],
                    code_attempts=code_attempts[:16],
                    code_uncertain_indices=code.uncertain_indices if code.text else [],
                    overlay_base64=overlay,
                    **self._provider_debug_payload(),
                ),
            )
        except Exception as ex:
            return AnalyzeResponse(
                qr_found=False,
                qr_content=None,
                wb_above_qr=False,
                code_below_qr=None,
                confidence=0.0,
                debug=DebugPayload(reason=f'Processing error: {type(ex).__name__}', **self._provider_debug_payload()),
            )

    def _provider_debug_payload(self) -> dict:
        payload = {
            'ocr_provider': 'local',
            'mistral_model': None,
            'mistral_request_id': None,
            'mistral_status_code': None,
            'mistral_error': None,
        }

        provider = self.primary_ocr
        if provider.__class__.__name__ == 'MistralAIProvider':
            payload['ocr_provider'] = 'mistral_ai'
            payload['mistral_model'] = getattr(provider, 'model', None)
            if hasattr(provider, 'get_last_call'):
                try:
                    info = provider.get_last_call()
                    payload['mistral_request_id'] = info.get('request_id')
                    payload['mistral_status_code'] = info.get('status_code')
                    payload['mistral_error'] = info.get('error')
                except Exception:
                    pass
        return payload
    def _select_best_orientation(self, image: np.ndarray) -> tuple[Optional[OrientationCandidate], list[str]]:
        if not settings.auto_rotate_enabled:
            detection = self._detect_largest_qr(image)
            if detection is None:
                return None, []
            rectified, above_roi, below_roi, above_bbox, below_bbox = self._rectify_and_extract_regions(image, detection.points)
            score, _, _, _ = self._orientation_score(above_roi, below_roi)
            selected = OrientationCandidate(
                angle=0,
                score=score,
                detection=detection,
                rectified=rectified,
                above_roi=above_roi,
                below_roi=below_roi,
                above_bbox=above_bbox,
                below_bbox=below_bbox,
            )
            return selected, [f'0:{score:.3f}']

        candidates: list[OrientationCandidate] = []
        score_labels: list[str] = []
        for angle in (0, 90, 180, 270):
            rotated = self._rotate_image(image, angle)
            detection = self._detect_largest_qr(rotated)
            if detection is None:
                score_labels.append(f'{angle}:no-qr')
                continue

            try:
                rectified, above_roi, below_roi, above_bbox, below_bbox = self._rectify_and_extract_regions(rotated, detection.points)
            except Exception:
                score_labels.append(f'{angle}:rectify-fail')
                continue

            score, red_score, code_score, contrast_score = self._orientation_score(above_roi, below_roi)
            score_labels.append(
                f'{angle}:{score:.3f}(red={red_score:.3f},code={code_score:.3f},delta={contrast_score:.3f})'
            )

            candidate = OrientationCandidate(
                angle=angle,
                score=score,
                detection=detection,
                rectified=rectified,
                above_roi=above_roi,
                below_roi=below_roi,
                above_bbox=above_bbox,
                below_bbox=below_bbox,
            )
            candidates.append(candidate)

            # Quick accept for already-correct images to keep latency low.
            if angle == 0 and red_score >= 0.008 and code_score >= 0.26:
                return candidate, score_labels

        if not candidates:
            return None, score_labels

        best = max(candidates, key=lambda c: c.score)
        return best, score_labels

    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        if angle == 0:
            return image
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def _orientation_score(self, above_roi: np.ndarray, below_roi: np.ndarray) -> tuple[float, float, float, float]:
        red_score = self._red_ratio(above_roi)
        code_score, below_dark = self._code_band_score(below_roi)
        _, above_dark = self._code_band_score(above_roi)
        contrast_score = max(0.0, below_dark - above_dark)
        score = red_score * 2.8 + code_score * 1.8 + contrast_score * 1.1
        return score, red_score, code_score, contrast_score

    def _red_ratio(self, roi: np.ndarray) -> float:
        if roi is None or roi.size == 0:
            return 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (140, 40, 40), (179, 255, 255))
        mask2 = cv2.inRange(hsv, (0, 40, 40), (12, 255, 255))
        red = cv2.bitwise_or(mask1, mask2)

        h = red.shape[0]
        y0 = int(h * 0.18)
        y1 = int(h * 0.92)
        crop = red[y0:y1, :]
        if crop.size == 0:
            return 0.0

        return float(np.count_nonzero(crop)) / float(crop.size)

    def _code_band_score(self, roi: np.ndarray) -> tuple[float, float]:
        if roi is None or roi.size == 0:
            return 0.0, 0.0

        h, w = roi.shape[:2]
        x0 = int(w * 0.08)
        x1 = max(x0 + 1, int(w * 0.92))
        y1 = max(1, int(h * 0.68))
        crop = roi[:y1, x0:x1]
        if crop.size == 0:
            return 0.0, 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        dark_ratio = float(np.count_nonzero(inv)) / float(inv.size)
        row_sum = (inv > 0).sum(axis=1)
        active_rows = float(np.count_nonzero(row_sum > int(inv.shape[1] * 0.08))) / max(1.0, float(inv.shape[0]))

        score = min(1.0, dark_ratio * 2.8) * 0.55 + min(1.0, active_rows * 3.2) * 0.45
        return score, dark_ratio

    def _detect_wb_above_qr(self, above_roi: np.ndarray) -> tuple[OCRResult, list[str]]:
        roi_candidates = self._build_wb_roi_candidates(above_roi)

        best = OCRResult(text='', confidence=0.0)
        attempts: list[str] = []
        for roi in roi_candidates:
            for prepared in self._build_wb_variants(roi):
                result = self._read_with_optional_fallback(
                    roi=prepared,
                    primary_reader=self.primary_ocr.read_wb,
                    fallback_reader=self.fallback_ocr.read_wb if self.fallback_ocr else None,
                )

                txt = (result.text or '').strip()
                if txt:
                    attempts.append(txt)

                if self._is_wb_match(result.text):
                    return OCRResult(text='WB', confidence=min(1.0, result.confidence + 0.1)), attempts

                if result.confidence > best.confidence:
                    best = result

        return best, attempts
    def _detect_code_below_qr(self, below_roi: np.ndarray) -> tuple[LineRead, list[str]]:
        if below_roi is None or below_roi.size == 0:
            return LineRead(text='', confidence=0.0, uncertain_indices=[]), []

        attempts: list[str] = []
        lines = self._split_code_lines(below_roi)

        line1 = self._best_line_read(lines[0], expected_len=7, tag='L1', attempts=attempts)
        line2 = self._best_line_read(lines[1], expected_len=4, tag='L2', attempts=attempts)
        line1 = LineRead(
            text=self._fit_line_length(line1.text, 7),
            confidence=line1.confidence,
            uncertain_indices=[i for i in line1.uncertain_indices if i < 7],
        )
        line2 = LineRead(
            text=self._fit_line_length(line2.text, 4),
            confidence=line2.confidence,
            uncertain_indices=[i for i in line2.uncertain_indices if i < 4],
        )

        if line1.text and line2.text:
            if line1.text == line2.text or line2.text in line1.text:
                final_text, final_uncertain = self._finalize_joined_code(line1.text, line1.uncertain_indices)
                return LineRead(text=final_text, confidence=line1.confidence, uncertain_indices=final_uncertain), attempts
            if line1.text in line2.text:
                final_text, final_uncertain = self._finalize_joined_code(line2.text, line2.uncertain_indices)
                return LineRead(text=final_text, confidence=line2.confidence, uncertain_indices=final_uncertain), attempts

            joined = f'{line1.text}{line2.text}'
            joined_uncertain = line1.uncertain_indices + [len(line1.text) + i for i in line2.uncertain_indices]
            conf = (line1.confidence + line2.confidence) / 2.0
            final_text, final_uncertain = self._finalize_joined_code(joined, joined_uncertain)
            return LineRead(text=final_text, confidence=conf, uncertain_indices=final_uncertain), attempts

        best_single = line1 if line1.confidence >= line2.confidence else line2
        if best_single.text:
            final_text, final_uncertain = self._finalize_joined_code(best_single.text, best_single.uncertain_indices)
            return LineRead(text=final_text, confidence=best_single.confidence, uncertain_indices=final_uncertain), attempts

        return LineRead(text='', confidence=0.0, uncertain_indices=[]), attempts
    def _best_line_read(self, line_roi: np.ndarray, expected_len: int, tag: str, attempts: list[str]) -> LineRead:
        best = LineRead(text='', confidence=0.0, uncertain_indices=[])
        best_score = -1.0
        best_digit_ratio = -1.0
        candidates: list[tuple[str, float, float, float]] = []

        variants = self._build_code_variants(line_roi)
        if expected_len == 4:
            variants.extend(self._build_line2_fast_variants(line_roi))

        for prepared in variants:
            result = self._read_with_optional_fallback(
                roi=prepared,
                primary_reader=self.primary_ocr.read_code,
                fallback_reader=self.fallback_ocr.read_code if self.fallback_ocr else None,
            )
            text = self._normalize_code_result(result.text)
            text = self._numeric_bias_normalize(text, expected_len)
            if text:
                attempts.append(f'{tag}:{text}')

            score = self._score_line_candidate(text, result.confidence, expected_len)
            digit_ratio = sum(ch.isdigit() for ch in text) / max(1, len(text)) if text else 0.0
            if text:
                candidates.append((text, result.confidence, score, digit_ratio))

            if (
                score > best_score
                or (abs(score - best_score) <= 0.08 and digit_ratio > best_digit_ratio + 0.05)
                or (abs(score - best_score) <= 0.02 and abs(digit_ratio - best_digit_ratio) <= 0.05 and result.confidence > best.confidence)
            ):
                best_score = score
                best_digit_ratio = digit_ratio
                best = LineRead(text=text, confidence=result.confidence, uncertain_indices=[])

            if (
                text
                and len(text) == expected_len
                and digit_ratio >= 0.85
                and result.confidence >= 0.82
            ):
                return LineRead(text=text, confidence=result.confidence, uncertain_indices=[])

        merged_text, merged_uncertain = self._merge_line_candidates(candidates, expected_len)
        if merged_text:
            merged_score = self._score_line_candidate(merged_text, best.confidence, expected_len)
            if merged_score >= best_score - 0.02:
                best = LineRead(text=merged_text, confidence=best.confidence, uncertain_indices=merged_uncertain)

        return best
    def _merge_line_candidates(self, candidates: list[tuple[str, float, float, float]], expected_len: int) -> tuple[str, list[int]]:
        if not candidates:
            return '', []

        ranked = sorted(candidates, key=lambda c: (c[2], c[1]), reverse=True)[:4]
        if expected_len in (4, 7):
            preferred_len = expected_len
        else:
            preferred_len = expected_len if any(len(t[0]) == expected_len for t in ranked) else len(ranked[0][0])
        preferred_len = max(1, min(18, preferred_len))

        out: list[str] = []
        uncertain: list[int] = []
        for i in range(preferred_len):
            bucket: dict[str, float] = {}
            for text, conf, score, _ in ranked:
                if i >= len(text):
                    continue
                ch = text[i]
                weight = 0.75 + max(0.0, score) + conf
                if expected_len in (4, 7) and ch.isdigit():
                    weight += 0.25
                bucket[ch] = bucket.get(ch, 0.0) + weight

            if not bucket:
                continue

            ordered = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)
            best_char, best_weight = ordered[0]
            second_weight = ordered[1][1] if len(ordered) > 1 else 0.0
            out.append(best_char)

            margin = best_weight - second_weight
            if margin < 0.8 or (len(ordered) > 1 and second_weight > best_weight * 0.35):
                uncertain.append(len(out) - 1)

        merged = ''.join(out)
        merged = self._numeric_bias_normalize(merged, expected_len)
        if len(merged) > 18:
            merged = merged[:18]
            uncertain = [i for i in uncertain if i < len(merged)]
        return merged, uncertain
    def _finalize_joined_code(self, text: str, uncertain_indices: list[int]) -> tuple[str, list[int]]:
        if not text:
            return '', []

        cleaned = self._normalize_code_result(text)
        uncertainty = [i for i in uncertain_indices if 0 <= i < len(cleaned)]
        if not cleaned:
            return '', []

        digit_ratio = sum(ch.isdigit() for ch in cleaned) / max(1, len(cleaned))
        if digit_ratio >= 0.82 and len(cleaned) > 11:
            best_window = cleaned[:11]
            best_start = 0
            best_score = -1.0
            for start in range(0, len(cleaned) - 10):
                window = cleaned[start:start + 11]
                ratio = sum(ch.isdigit() for ch in window) / 11.0
                unsure_penalty = sum(1 for idx in uncertainty if start <= idx < start + 11) * 0.08
                score = ratio * 2.0 - unsure_penalty
                if score > best_score:
                    best_score = score
                    best_window = window
                    best_start = start
            cleaned = best_window
            uncertainty = [idx - best_start for idx in uncertainty if best_start <= idx < best_start + 11]

        if digit_ratio >= 0.82 and len(cleaned) == 11:
            chars = list(cleaned)
            for i, ch in enumerate(chars):
                if ch == 'B':
                    chars[i] = '8'
                elif ch == 'A':
                    chars[i] = '4'
                elif ch == 'G':
                    chars[i] = '6'
            cleaned = ''.join(chars)

        return cleaned, sorted(set(uncertainty))

    def _fit_line_length(self, text: str, expected_len: int) -> str:
        cleaned = self._normalize_code_result(text)
        if not cleaned:
            return ''
        if len(cleaned) == expected_len:
            return cleaned
        if len(cleaned) < expected_len:
            return cleaned

        # Most noisy reads add one leading character from nearby barcode/edge.
        if len(cleaned) == expected_len + 1 and cleaned[0] in {'0', 'O', 'Q', 'D'}:
            trimmed = cleaned[1:]
            if len(trimmed) == expected_len:
                return trimmed

        best = cleaned[:expected_len]
        best_score = -1.0
        center = (len(cleaned) - expected_len) / 2.0
        for start in range(0, len(cleaned) - expected_len + 1):
            window = cleaned[start : start + expected_len]
            digit_ratio = sum(ch.isdigit() for ch in window) / max(1, expected_len)
            non_zero_bonus = 0.15 if window and window[0] != '0' else 0.0
            center_penalty = abs(start - center) * 0.06
            score = digit_ratio * 1.8 + non_zero_bonus - center_penalty
            if score > best_score:
                best_score = score
                best = window
        return best

    def _split_code_lines(self, below_roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = below_roi.shape[:2]
        band = below_roi[: int(h * 0.62), :]

        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        _, inv = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        row_sum = (inv > 0).sum(axis=1)
        threshold = max(10, int(inv.shape[1] * 0.06))
        active = row_sum > threshold

        segments: list[tuple[int, int]] = []
        start = None
        for i, flag in enumerate(active):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start >= 6:
                    segments.append((start, i - 1))
                start = None
        if start is not None:
            segments.append((start, len(active) - 1))

        if len(segments) >= 2:
            segments = sorted(segments, key=lambda s: s[0])[:2]

            def to_crop(seg: tuple[int, int]) -> np.ndarray:
                y0 = max(0, int(seg[0] / 2) - 2)
                y1 = min(band.shape[0], int(seg[1] / 2) + 3)
                x0 = int(w * 0.08)
                x1 = int(w * 0.92)
                return band[y0:y1, x0:x1]

            return to_crop(segments[0]), to_crop(segments[1])

        bh = band.shape[0]
        line1 = band[: int(bh * 0.52), :]
        line2 = band[int(bh * 0.40) :, :]
        return line1, line2

    def _build_wb_roi_candidates(self, above_roi: np.ndarray) -> list[np.ndarray]:
        if above_roi is None or above_roi.size == 0:
            return []

        h, _ = above_roi.shape[:2]
        low_band_start = int(h * 0.42)
        return [
            above_roi,
            above_roi[low_band_start:h, :],
        ]

    def _build_wb_variants(self, roi: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (140, 40, 40), (179, 255, 255))
        mask2 = cv2.inRange(hsv, (0, 40, 40), (12, 255, 255))
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_up = cv2.resize(red_mask, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_NEAREST)

        return [
            roi,
            cv2.cvtColor(th, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(red_up, cv2.COLOR_GRAY2BGR),
        ]

    def _build_code_variants(self, roi: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=2.7, fy=2.7, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        variants = [
            roi,
            cv2.cvtColor(th, cv2.COLOR_GRAY2BGR),
        ]
        if not settings.ocr_fast_mode:
            _, thi = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            variants.append(cv2.cvtColor(thi, cv2.COLOR_GRAY2BGR))
        return variants
    def _build_line2_fast_variants(self, roi: np.ndarray) -> list[np.ndarray]:
        if roi is None or roi.size == 0:
            return []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        up = cv2.resize(gray, None, fx=2.8, fy=2.8, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(up, (3, 3), 0)
        _, th_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        th = cv2.bitwise_not(th_inv)
        return [cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)]

    def _normalize_code_result(self, text: str) -> str:
        if not text:
            return ''

        out = re.sub(r'[^A-Z0-9]', '', text.upper())
        out = out.replace('O', '0').replace('I', '1')

        if len(out) > 16:
            groups = re.findall(r'[A-Z0-9]{3,12}', out)
            out = groups[0] if groups else out[:12]

        return out

    def _numeric_bias_normalize(self, text: str, expected_len: int) -> str:
        if not text:
            return ''

        n = len(text)
        digit_ratio = sum(ch.isdigit() for ch in text) / max(1, n)
        if digit_ratio < 0.7:
            return text

        chars = list(text)
        base_map = {
            'B': '8',
            'S': '5',
            'Z': '2',
            'O': '0',
            'Q': '0',
            'D': '0',
            'I': '1',
            'L': '1',
            'T': '1',
        }

        for i, ch in enumerate(chars):
            if ch in base_map:
                chars[i] = base_map[ch]
                continue

            if ch == 'A':
                prev_digit = i > 0 and chars[i - 1].isdigit()
                next_digit = i < n - 1 and chars[i + 1].isdigit()
                if prev_digit and next_digit:
                    chars[i] = '1'
                elif i == 0 and n >= 2 and (chars[1].isdigit() or chars[1] in {'8', 'B'}):
                    chars[i] = '4'

        out = ''.join(chars)
        if expected_len in (4, 7) and sum(ch.isdigit() for ch in out) >= max(3, expected_len - 1):
            return out
        return out

    def _score_line_candidate(self, code: str, confidence: float, expected_len: int) -> float:
        if not code:
            return -1.0

        n = len(code)
        digit_ratio = sum(ch.isdigit() for ch in code) / max(1, n)
        len_score = 1.0 - min(abs(n - expected_len), expected_len) / max(1, expected_len)
        return digit_ratio * 1.5 + len_score + confidence

    def _is_wb_match(self, text: str) -> bool:
        if not text:
            return False

        normalized = text.upper().strip()
        normalized = re.sub(r'[^A-Z0-9]', '', normalized)
        normalized = normalized.replace('VV', 'W').replace('VVB', 'WB').replace('W8', 'WB').replace('WO', 'WB').replace('W0', 'WB')

        return normalized == 'WB' or 'WB' in normalized

    def _read_with_optional_fallback(self, roi: np.ndarray, primary_reader, fallback_reader) -> OCRResult:
        primary = primary_reader(roi)
        if fallback_reader is None:
            return primary
        if primary.confidence >= settings.local_confidence_threshold:
            return primary

        fallback = fallback_reader(roi)
        if fallback.confidence > primary.confidence:
            return fallback
        return primary
    def _detect_largest_qr(self, image: np.ndarray) -> Optional[QRDetection]:
        detect_img, scale_back = self._prepare_qr_detection_image(image)
        candidates: list[tuple[np.ndarray, Optional[str]]] = []

        try:
            multi_out = self.qr_detector.detectAndDecodeMulti(detect_img)
            if isinstance(multi_out, tuple) and len(multi_out) >= 3:
                found = bool(multi_out[0])
                decoded_info = multi_out[1] if len(multi_out) > 1 else []
                points = multi_out[2]
                if found and points is not None:
                    for idx, quad in enumerate(points):
                        prepared = self._normalize_quad(quad)
                        if prepared is not None:
                            prepared = prepared * scale_back
                            content = None
                            if isinstance(decoded_info, (list, tuple)) and idx < len(decoded_info):
                                raw = decoded_info[idx]
                                content = str(raw).strip() if raw is not None else None
                                if content == '':
                                    content = None
                            candidates.append((prepared, content))
        except Exception:
            pass

        if not candidates:
            try:
                decoded, single_points, _ = self.qr_detector.detectAndDecode(detect_img)
                if single_points is not None and len(single_points):
                    prepared = self._normalize_quad(single_points[0])
                    if prepared is not None:
                        prepared = prepared * scale_back
                        content = decoded.strip() if isinstance(decoded, str) and decoded.strip() else None
                        candidates.append((prepared, content))
            except Exception:
                pass

        if not candidates:
            try:
                _, single_points = self.qr_detector.detect(detect_img)
                if single_points is not None and len(single_points):
                    prepared = self._normalize_quad(single_points[0])
                    if prepared is not None:
                        prepared = prepared * scale_back
                        candidates.append((prepared, None))
            except Exception:
                pass

        if not candidates:
            return None

        largest_points, largest_content = max(candidates, key=lambda c: self._polygon_area(c[0]))
        x, y, w, h = cv2.boundingRect(largest_points.astype(np.float32))
        if w <= 0 or h <= 0:
            return None
        return QRDetection(
            points=largest_points,
            bbox=BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h)),
            content=largest_content,
        )

    def _prepare_qr_detection_image(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        max_side = max(h, w)
        target_max_side = 1700
        if max_side <= target_max_side:
            return image, 1.0

        scale = target_max_side / float(max_side)
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, (1.0 / scale)

    def _normalize_quad(self, quad: np.ndarray) -> Optional[np.ndarray]:
        arr = np.array(quad, dtype=np.float32).reshape(-1, 2)
        if arr.shape != (4, 2):
            return None
        if not np.isfinite(arr).all():
            return None
        if self._polygon_area(arr) < 64.0:
            return None
        return arr

    def _rectify_and_extract_regions(
        self,
        image: np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, BoundingBox, BoundingBox]:
        ordered = self._order_points(points)
        edge = int(max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1])))
        side = max(120, min(900, edge))

        canvas_w = side * 3
        canvas_h = side * 4
        qr_top = side
        qr_bottom = side * 2

        dst = np.array(
            [
                [side, qr_top],
                [2 * side, qr_top],
                [2 * side, qr_bottom],
                [side, qr_bottom],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        rectified = cv2.warpPerspective(image, matrix, (canvas_w, canvas_h))
        if rectified is None or rectified.size == 0:
            raise ValueError('Could not rectify QR region')

        above = rectified[0:qr_top, side : 2 * side]
        below = rectified[qr_bottom : min(canvas_h, qr_bottom + int(side * 1.85)), side : 2 * side]

        above_bbox = BoundingBox(x=side, y=0, w=side, h=qr_top)
        below_bbox = BoundingBox(x=side, y=qr_bottom, w=side, h=below.shape[0])
        return rectified, above, below, above_bbox, below_bbox
    def _build_overlay(self, rectified: np.ndarray) -> str:
        if rectified is None or rectified.size == 0:
            return ''

        overlay = rectified.copy()
        side = rectified.shape[1] // 3
        qr_top = side
        qr_bottom = side * 2
        below_bottom = min(rectified.shape[0] - 1, qr_bottom + int(side * 1.85))

        cv2.rectangle(overlay, (side, qr_top), (2 * side, qr_bottom), (0, 255, 0), 2)
        cv2.rectangle(overlay, (side, 0), (2 * side, qr_top), (255, 180, 0), 2)
        cv2.rectangle(overlay, (side, qr_bottom), (2 * side, below_bottom), (0, 180, 255), 2)

        ok, encoded = cv2.imencode('.jpg', overlay)
        if not ok:
            return ''
        return base64.b64encode(encoded.tobytes()).decode('ascii')
    def _polygon_area(self, points: np.ndarray) -> float:
        x = points[:, 0]
        y = points[:, 1]
        return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.astype(np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)



































