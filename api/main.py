from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

# 使用 ROOT 變數組合路徑，增加部署靈活性
ROOT = Path(__file__).resolve().parents[1]
WEIGHT_PATH = ROOT / "weights" / "Proposed" / "best.pt"

app = FastAPI(title="UAV-YOLOv11 小物件偵測")
model = YOLO(str(WEIGHT_PATH))

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    """
    偵測並回傳標註後的圖片 (Ultralytics UI)
    """
    # 讀取圖片
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 執行推理
    results = model.predict(img, conf=0.25)
    
    # 利用 Ultralytics 內建的 .plot() 繪圖
    # im_bgr 會是一個 numpy array，風格與官方完全一致
    annotated_frame = results[0].plot(line_width=1, font_size=1.5) 
    
    # 將 BGR 轉回 RGB 並編碼為 JPEG
    _, im_encode = cv2.imencode(".jpg", annotated_frame)
    return StreamingResponse(io.BytesIO(im_encode.tobytes()), media_type="image/jpeg")

@app.post("/predict_json")
async def predict_json(file: UploadFile = File(...)):
    """
    JSON 回傳邏輯，保留給後端數據處理使用
    """
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    results = model.predict(img, conf=0.25)
    
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]
            })
    return {"count": len(detections), "results": detections}