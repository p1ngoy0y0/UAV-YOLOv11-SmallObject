from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.detector import Detector
from api import ROOT

# 使用 Lifespan 管理模型載入，避免重複載入佔用顯存
detector = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時載入模型
    weight_path = ROOT / "weights" / "Proposed" / "best.pt"
    detector["main"] = Detector(weight_path)
    yield
    # 關閉時清理資源 (如果需要)
    detector.clear()

app = FastAPI(
    title="UAV-YOLOv11 空拍圖小物件偵測",
    description="基於 BiFPN 與 SPD-Conv 之改良 YOLOv11",
    lifespan=lifespan
)