import numpy as np
from ultralytics import YOLO
from pathlib import Path

class Detector:
    def __init__(self, weight_path: Path):
        # 確保路徑是字串格式
        self.model = YOLO(str(weight_path))

    def predict(self, img_bgr: np.ndarray, conf: float = 0.25):
        # 執行推理
        results = self.model.predict(img_bgr, conf=conf)
        return results

    def plot_detection(self, result):
        # 利用 Ultralytics 內建繪圖
        return result.plot(line_width=1, font_size=1.5)

    def get_json_results(self, result):
        detections = []
        for box in result.boxes:
            detections.append({
                "class": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]
            })
        return detections