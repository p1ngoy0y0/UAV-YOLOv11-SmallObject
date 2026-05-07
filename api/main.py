import io
import cv2
import numpy as np
from fastapi import File, UploadFile
from fastapi.responses import StreamingResponse
from api.app import app, detector

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 呼叫 detector 引擎
    results = detector["main"].predict(img)
    annotated_frame = detector["main"].plot_detection(results[0])
    
    _, im_encode = cv2.imencode(".jpg", annotated_frame)
    return StreamingResponse(io.BytesIO(im_encode.tobytes()), media_type="image/jpeg")

@app.post("/predict_json")
async def predict_json(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = detector["main"].predict(img)
    detections = detector["main"].get_json_results(results[0])
    
    return {"count": len(detections), "results": detections}