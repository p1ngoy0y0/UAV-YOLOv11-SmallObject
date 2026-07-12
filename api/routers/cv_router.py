import io
import json
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import StreamingResponse 

# 物理邏輯：同一個產線（cv）下的不同功能分支
router = APIRouter(prefix="/cv", tags=["YOLOv11 Detector"])

@router.post("/predict_image")
async def predict_image(request: Request, file: UploadFile = File(...)):
    # ⚡ 執行期動態獲取全域單例物件，避開硬碟 I/O，直擊 VRAM 指針
    detector = request.app.state.models["yolo"]

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = detector.predict(img)
    annotated_frame = detector.plot_detection(results[0])

    _, im_encode = cv2.imencode(".jpg", annotated_frame)
    return StreamingResponse(io.BytesIO(im_encode.tobytes()), media_type="image/jpeg")

@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    detector = request.app.state.models["yolo"]

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = detector.predict(img)
    detections = detector.get_json_results(results[0])

    # 統計邏輯優化
    class_counts = {}
    for obj in detections:
        cls_name = obj["class"]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    return {
        "count": len(detections), 
        "results": class_counts,
    }