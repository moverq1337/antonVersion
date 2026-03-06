from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import AnalyzeResponse
from app.services.analyzer import ImageAnalyzer
from app.services.ocr import build_mistral_ai_provider

app = FastAPI(title='Mistral QR WB Analyzer', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

mistral_ai_provider = build_mistral_ai_provider()
mistral_ai_analyzer = ImageAnalyzer(primary_ocr=mistral_ai_provider, fallback_ocr=None) if mistral_ai_provider else None


@app.get('/')
def root() -> FileResponse:
    return FileResponse('static/index.html')


@app.post('/analyze', response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    require_wb: bool = Form(True),
) -> AnalyzeResponse:
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Only image uploads are supported')

    if mistral_ai_analyzer is None:
        raise HTTPException(status_code=400, detail='Mistral AI is not configured (check settings_ai)')

    payload = await file.read()
    img_array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail='Could not decode image')

    try:
        return mistral_ai_analyzer.analyze(image, require_wb=require_wb)
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f'Image processing failed: {type(ex).__name__}') from ex


app.mount('/static', StaticFiles(directory='static'), name='static')
