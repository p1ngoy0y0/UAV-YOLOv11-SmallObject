from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import io
from PIL import Image
import numpy as np

# 1. 初始化 FastAPI APP
app = FastAPI(title="UAV-YOLOv11 空拍圖小物件偵測")

# 2. 載入權重
model = YOLO("/home/r11525124/MASTER/Portfolio/weights/Proposed/best.pt") 

@app.get("/")
async def root():
    return {"message": "UAV-YOLOv11 API 就緒"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 讀取上傳的圖片內容
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
    
    # 3. 執行推理
    results = model.predict(img, conf=0.25)
    
    # 4. 整理回傳結果
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0] # [x1, y1, x2, y2]
            })
            
    return {"count": len(detections), "results": detections}